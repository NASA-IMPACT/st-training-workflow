import json
import os
import re

import pandas as pd
import seaborn as sns
import sentence_transformers.util as util
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import NanoBEIREvaluator

# Define the device to use
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
ks = [1, 3, 5, 10]
json_output_path = "results_json/nanobeir_eval_dump.json"

models = {
    "modernbert-embed-base": "nomic-ai/modernbert-embed-base",
    "nasa-smd-ibm-st-v2": "nasa-impact/nasa-smd-ibm-st-v2",
    "indus-sde-st-v0.1": "nasa-impact/indus-sde-st-v0.1",
    "indus-sde-st-v0.2_whole-moon-14": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/model-qr3ln5om:v1/checkpoint-116000",
}


# set datasets to None for evalauting on all datasets
# datasets = ["QuoraRetrieval", "MSMARCO"]
datasets = None

evaluator = NanoBEIREvaluator(
    dataset_names=datasets,
    mrr_at_k=ks,
    accuracy_at_k=ks,
    precision_recall_at_k=ks,
    map_at_k=ks,
    ndcg_at_k=ks,
    show_progress_bar=True,
    batch_size=32,
    write_csv=True,
)

# check if the json_output_path file exists
# if it does, load it to all_results else initilize an empty dictionary
if os.path.exists(json_output_path):
    # load the json
    with open(json_output_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)
else:
    all_results = {}


for model_name, model_path in models.items():
    print(f"Evaluating model: {model_name}")
    if model_name in all_results:
        print(f"Model {model_name} already evaluated. Skipping...")
        continue

    model = SentenceTransformer(model_path)
    results = evaluator(model)
    all_results[model_name] = results

    print(results)

    # Clean up
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


# before we save the json, we need to add mean for each dataset
print("Adding mean results for each model...")
for model_name in all_results:
    metric_names = set([i.split("_")[-1] for i in list(all_results[model_name].keys())])
    dataset_names = set(
        [
            "_".join(i.split("_")[:-2])
            for i in all_results[model_name]
            if i.split("_")[-3] != "mean"
        ],
    )
    mean_result = {}

    # Calculate the mean for each metric

    for metric in metric_names:
        values = []
        for dataset_n in dataset_names:
            key_name = f"{dataset_n}_cosine_{metric}"
            values.append(all_results[model_name][key_name])

        mean_result[f"NanoBEIR_mean_cosine_{metric}"] = sum(values) / (
            len(values) if len(values) > 0 else 1
        )

    all_results[model_name] = {**all_results[model_name], **mean_result}


with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)


# Plotting the results
desired_metric_types = ["mrr", "accuracy"]  # Changed to MRR and Accuracy
output_dir_plots = "results_plots/nanobeir_eval_plots"


os.makedirs(output_dir_plots, exist_ok=True)  # Ensure the output directory exists
# --- Data Transformation ---
# This section converts the nested dictionary into a flat list of records,
# which is then converted into a Pandas DataFrame for easier manipulation.
parsed_data = []
for model_name, metrics_data in all_results.items():
    for metric_key, value in metrics_data.items():
        try:
            # Use regex to extract dataset, metric type, and k value
            # This makes the parsing more robust to variations in the key format
            match = re.match(r"^(.*?)_cosine_([a-zA-Z]+)@(\d+)$", metric_key)
            if not match:
                print(f"Warning: Unexpected format for key: {metric_key}. Skipping.")
                continue

            dataset_name = match.group(1)
            metric_type = match.group(2)
            k = int(match.group(3))  # Convert k to an integer

            # Append the parsed data as a dictionary to the list
            parsed_data.append(
                {
                    "model": model_name,
                    "dataset": dataset_name,
                    "metric_type": metric_type,
                    "k": k,
                    "value": value,
                },
            )
        except Exception as e:
            # Catch any other parsing errors and report them.
            print(f"Error parsing key '{metric_key}': {e}. Skipping.")

# Convert the list of dictionaries into a Pandas DataFrame
df = pd.DataFrame(parsed_data)

# Convert 'k' to a string type for plotting.
# This ensures that 'k' is treated as a categorical variable on the x-axis,
# preventing Seaborn from trying to interpret it as a continuous number and aggregate.
df["k_str"] = df["k"].astype(str)

# --- Plotting Setup ---
# Get unique identifiers for models, datasets, and metric types present in the data.
# These are now dynamically determined from the parsed DataFrame.
unique_datasets = df["dataset"].unique()
# Filter metric types to only include 'recall' and 'mrr'

unique_metric_types = [
    mt for mt in df["metric_type"].unique() if mt in desired_metric_types
]

unique_models = df["model"].unique()
unique_k_values = df["k"].unique()  # Get all unique k values from the data

# Define a consistent color palette for models.
# This ensures that each model has the same color across all plots for easy comparison.
colors = sns.color_palette("tab10", len(unique_models))
model_color_map = {model: colors[i] for i, model in enumerate(unique_models)}

# --- Generate Plots for Each Dataset ---
# Iterate through each unique dataset to create a separate figure.
for dataset in unique_datasets:
    # Create a figure with subplots. There will be one row and as many columns
    # as there are unique metric types (e.g., MRR and Accuracy).
    # sharey=False is important as their value ranges might differ.
    fig, axes = plt.subplots(1, len(unique_metric_types), figsize=(14, 6), sharey=False)

    # Set the main title for the entire figure, indicating the current dataset.
    # Replace underscores for better readability.
    fig.suptitle(
        f"Model Performance on {dataset.replace('_', ' ').title()}",
        fontsize=16,
        y=1.05,
    )  # Increased y to make space for bottom legend

    # Ensure 'axes' is always an array, even if there's only one metric type,
    # so that indexing `axes[i]` works consistently.
    if len(unique_metric_types) == 1:
        axes = [axes]

    # Iterate through each metric type (e.g., 'mrr', 'accuracy') to create a subplot.
    for i, metric_type in enumerate(unique_metric_types):
        ax = axes[i]  # Get the current subplot axis

        # Filter the DataFrame to get data relevant to the current dataset and metric type.
        plot_data = df[
            (df["dataset"] == dataset) & (df["metric_type"] == metric_type)
        ].copy()

        # Sort the data by 'k' value to ensure the bars are ordered correctly on the x-axis.
        plot_data = plot_data.sort_values(by="k")

        # Create the bar plot using Seaborn.
        # x='k_str' uses the categorical string representation of k.
        # hue='model' groups bars by model and uses the model_color_map for consistent coloring.
        sns.barplot(
            data=plot_data,
            x="k_str",  # Use string for categorical x-axis
            y="value",
            hue="model",
            palette=model_color_map,  # Apply consistent color map
            ax=ax,
        )

        # Add value labels on top of each bar
        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%.2f",
                fontsize=8,
                label_type="edge",
                padding=3,
            )

        # Set subplot title, x-axis label, and y-axis label.
        ax.set_title(f"{metric_type.capitalize()}@k")
        ax.set_xlabel("k")
        ax.set_ylabel(f"{metric_type.capitalize()} Value")

        # Set y-axis limits from 0 to 1, as values are typically between this range.
        ax.set_ylim(0, 1.0)

        # Add a grid to the y-axis for easier reading of values.
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # Remove individual subplot legends as we'll add a single figure-wide legend
        if ax.get_legend():
            ax.get_legend().remove()

    # Adjust the layout to make space for the legend at the bottom.
    plt.tight_layout(rect=[0, 0.1, 1, 0.98])  # Adjusted rect to make space at bottom

    # Create a single legend for the entire figure at the bottom
    handles, labels = axes[
        0
    ].get_legend_handles_labels()  # Get handles/labels from the first subplot
    fig.legend(
        handles,
        labels,
        title="Model",
        loc="lower center",
        ncol=len(unique_models),
        bbox_to_anchor=(0.5, 0.0),
        borderaxespad=0.0,
    )  # Legend at bottom, one row

    # Display the plot for the current dataset.
    plt.savefig(
        os.path.join(output_dir_plots, f"{dataset}_performance_plots.png"),
        bbox_inches="tight",
    )
    # plt.show()
