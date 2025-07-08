import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sentence_transformers.util as util
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

# Define the device to use
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
ks = [1, 3, 5, 10]
json_output_path = "results_json/nasa_ir_eval_dump.json"

models = {
    "modernbert-embed-base": "nomic-ai/modernbert-embed-base",
    "nasa-smd-ibm-st-v2": "nasa-impact/nasa-smd-ibm-st-v2",
    "indus-sde-st-v0.1": "nasa-impact/indus-sde-st-v0.1",
    "indus-sde-st-v0.2_whole-moon-14": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/model-qr3ln5om:v1/checkpoint-116000",
}

corpus = load_dataset(
    "nasa-impact/nasa-smd-IR-benchmark",
    data_files="https://huggingface.co/datasets/nasa-impact/nasa-smd-IR-benchmark/resolve/main/corpus.jsonl",
    split="train",
)
queries = load_dataset(
    "nasa-impact/nasa-smd-IR-benchmark",
    data_files="https://huggingface.co/datasets/nasa-impact/nasa-smd-IR-benchmark/resolve/main/queries.jsonl",
    split="train",
)
relevant_docs_data = load_dataset("nasa-impact/nasa-smd-IR-benchmark", split="test")


corpus = {row["_id"]: row["text"] for i, row in enumerate(corpus)}
queries = {row["_id"]: row["text"] for row in queries}
relevant_docs_data = (
    relevant_docs_data.to_pandas().groupby("query-id")["corpus-id"].apply(set).to_dict()
)
relevant_docs_data = {
    str(k): {str(item) for item in v} for k, v in relevant_docs_data.items()
}


# for testing only
# relevent_docs = set([str(i) for k, v in relevant_docs_data.items() for i in v])
# corpus = {k: v for k, v in corpus.items() if k in relevent_docs}

evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs_data,
    name="nasa_ir_evaluator",
    batch_size=8,
    mrr_at_k=ks,
    ndcg_at_k=ks,
    accuracy_at_k=ks,
    precision_recall_at_k=ks,
    map_at_k=ks,
    show_progress_bar=True,
    # convert_to_tensor=True,      # ensure corpus embeddings are torch.Tensors
    write_csv=True,
    # corpus_chunk_size=5000,
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

    # Ensure CUDA context is properly set
    if device.startswith("cuda"):
        torch.cuda.set_device(device)

    model = SentenceTransformer(model_path, device=device)

    # Double-check model is on correct device
    model = model.to(device)

    results = evaluator(model)
    all_results[model_name] = results

    print(results)

    # Clean up
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)


# now plotting
output_dir_plots = "results_plots/nasa_ir_eval_plots"

os.makedirs(output_dir_plots, exist_ok=True)  # Ensure the output directory exists
# Process the data into a list of records
records = []
for model, metrics in all_results.items():
    for metric_name_full, value in metrics.items():
        parts = metric_name_full.split("@")
        k = int(parts[1])
        metric_name = parts[0].split("_")[-1]
        records.append(
            {
                "model": model,
                "metric": metric_name,
                "k": k,
                "value": value,
            },
        )

# Create a pandas DataFrame
df = pd.DataFrame(records)

# Create the bar plot using seaborn's catplot for faceting
g = sns.catplot(
    data=df,
    x="k",
    y="value",
    hue="model",
    col="metric",
    kind="bar",
    col_wrap=3,
    sharey=False,
    height=4,
    aspect=1.5,
    legend_out=True,
)

# Customize the plot with titles and labels
g.fig.suptitle("Model Performance Comparison by Metric and K-Value On NASA IR", y=1.03)
g.set_titles("Metric: {col_name}")
g.set_axis_labels("K Value", "Score")
g.despine(left=True)

# --- NEW: Add value labels on top of each bar ---
for ax in g.axes.flat:
    # Iterate through the bars in each subplot
    for p in ax.patches:
        # Get the height of the bar and format it to 2 decimal places
        value = f"{p.get_height():.2f}"
        # Define the position for the annotation
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        # Add the text to the plot
        ax.annotate(
            value,
            (x, y),
            ha="center",
            va="center",
            xytext=(0, 5),  # 5 points vertical offset
            textcoords="offset points",
            fontsize=8,
        )

# Move and format the legend
sns.move_legend(
    g,
    "lower center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=4,
    title=None,
    frameon=False,
)

# Adjust the bottom of the figure to make space for the legend
g.fig.subplots_adjust(bottom=0.15)


# Display the plot
plt.savefig(
    os.path.join(output_dir_plots, f"NASA_IR_performance_plots.png"),
    bbox_inches="tight",
)
# plt.show()
