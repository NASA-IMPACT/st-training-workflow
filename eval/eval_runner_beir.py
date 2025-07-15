import json
import logging
import math
import os
import pathlib
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import torch._dynamo
from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader
from custum_evals import MultiGPUInformationRetrievalEvaluator
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.evaluation import InformationRetrievalEvaluator

torch._dynamo.config.suppress_errors = True


#### Just some setup for logging ####
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

json_output_path = "results_json/beir_eval_results/"
output_dir_plots = "results_plots/beir_eval_results"

os.makedirs(json_output_path, exist_ok=True)
os.makedirs(output_dir_plots, exist_ok=True)


split = "test"
ks = [1, 3, 5, 10]

models = {
    "modernbert-embed-base": "nomic-ai/modernbert-embed-base",
    "nasa-smd-ibm-st-v2": "nasa-impact/nasa-smd-ibm-st-v2",
    "indus-sde-st-v0.1": "nasa-impact/indus-sde-st-v0.1",
    "indus-sde-st-v0.2_whole-moon-14": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/model-qr3ln5om:v1/checkpoint-116000",
    "indus-sde-st-v0.2_atomic-plasma-15": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/model-ykf0bews:v1/checkpoint-13000",
    "indus-sde-st-v0.2_vocal-river-16": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/model-dexwvlkj:v1/checkpoint-13000",
}

_DATASETS = [
    "trec-covid",
    "nfcorpus",
    "nq",
    "hotpotqa",
    "fiqa",
    "arguana",
    "webis-touche2020",
    "dbpedia-entity",
    "scidocs",
    "fever",
    "climate-fever",
    "scifact",
    # "msmarco",
    # "quora",
    # "cqadupstack",
]

all_results = {}


def plot_beir_evaluation_results(
    path_to_json_files,
    output_folder_path,
    metrics_to_plot=None,
):
    """
    Generates subplot images from BEIR evaluation results and saves them.
    (This function is unchanged)
    """
    # ... (The code for the plotting function remains exactly the same as the last version) ...
    try:
        os.makedirs(output_folder_path, exist_ok=True)
        print(f"Plots will be saved to: {os.path.abspath(output_folder_path)}")
    except OSError as e:
        print(f"Error creating directory {output_folder_path}: {e}")
        return

    try:
        files = [f for f in os.listdir(path_to_json_files) if f.endswith(".json")]
    except FileNotFoundError:
        print(f"Error: The directory '{path_to_json_files}' was not found.")
        return

    for file in files:
        file_path = os.path.join(path_to_json_files, file)
        dataset_name = file.replace(".json", "")

        with open(file_path, "r") as f:
            data = json.load(f)

        records = []
        for model, metrics_data in data.items():
            for key, value in metrics_data.items():
                try:
                    metric_part = key.split("_cosine_")[-1]
                    metric_name, k = metric_part.split("@")
                    records.append(
                        {
                            "model": model,
                            "metric": metric_name,
                            "k": int(k),
                            "value": value,
                        },
                    )
                except (IndexError, ValueError):
                    print(
                        f"Warning: Could not parse metric key '{key}' in file '{file}'.",
                    )

        if not records:
            continue

        df = pd.DataFrame(records)

        print("Availab;e metrics in the dataset:", df["metric"].unique())

        available_metrics = sorted(df["metric"].unique())
        if metrics_to_plot is None:
            final_metrics_list = available_metrics
        else:
            final_metrics_list = [m for m in metrics_to_plot if m in available_metrics]

        if not final_metrics_list:
            continue

        num_metrics = len(final_metrics_list)
        ncols = 2 if num_metrics > 1 else 1
        nrows = math.ceil(num_metrics / ncols)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(14, 5 * nrows),
            constrained_layout=True,
        )

        flat_axes = np.array(axes).flatten()

        fig.suptitle(
            f"Evaluation for {dataset_name.upper()}",
            fontsize=20,
            weight="bold",
        )

        for i, metric in enumerate(final_metrics_list):
            ax = flat_axes[i]
            metric_df = df[df["metric"] == metric]

            pivot_df = metric_df.pivot(index="k", columns="model", values="value")

            if pivot_df.empty:
                ax.set_title(f"{metric.upper()}\n(No Data)", fontsize=14)
                continue

            pivot_df.plot(kind="bar", ax=ax, width=0.8, legend=False)

            ax.set_title(metric.upper(), fontsize=14)
            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.set_xlabel("Top-k", fontsize=12)
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            ax.tick_params(axis="x", rotation=0)

            for p in ax.patches:
                ax.annotate(
                    f"{p.get_height():.3f}",
                    (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center",
                    va="center",
                    fontsize=9,
                    xytext=(0, 9),
                    textcoords="offset points",
                )

        for i in range(num_metrics, len(flat_axes)):
            flat_axes[i].axis("off")

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=3,
            fancybox=True,
        )

        plot_filename = f"{dataset_name}_metrics_summary.png"
        full_save_path = os.path.join(output_folder_path, plot_filename)

        plt.savefig(full_save_path, bbox_inches="tight", pad_inches=0.5)
        print(f"✅ Successfully saved consolidated plot to '{full_save_path}'")
        plt.close(fig)


if __name__ == "__main__":
    # start to evaluate
    for dname in _DATASETS:

        if os.path.exists(f"{json_output_path}{dname}.json"):
            # load the json
            with open(f"{json_output_path}{dname}.json", "r", encoding="utf-8") as f:
                all_results[dname] = json.load(f)
        else:
            all_results[dname] = {}

        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dname}.zip"
        out_dir = os.path.join(pathlib.Path.cwd(), "datasets")
        data_path = beir_util.download_and_unzip(url, out_dir)

        # Load the dataset using the BEIR loader
        corpus_beir, queries, qrels_beir = GenericDataLoader(
            data_folder=data_path,
        ).load(
            split="test",
        )

        # The corpus needs to be flattened from Dict[str, Dict[str,str]] to Dict[str, str]
        # We'll concatenate the title and text for each document.
        corpus = {
            doc_id: (doc.get("title", "") + " " + doc.get("text", "")).strip()
            for doc_id, doc in corpus_beir.items()
        }

        # The qrels need to be converted to the relevant_docs format: Dict[str, Set[str]]
        # We'll only consider documents with a relevance score > 0.
        relevant_docs = defaultdict(set)
        for query_id, doc_scores in qrels_beir.items():
            for doc_id, score in doc_scores.items():
                if score > 0:
                    relevant_docs[query_id].add(doc_id)

        evaluator = MultiGPUInformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"{dname}_evaluator",
            batch_size=64,
            mrr_at_k=ks,
            ndcg_at_k=ks,
            accuracy_at_k=ks,
            precision_recall_at_k=ks,
            map_at_k=ks,
            show_progress_bar=True,
            write_csv=True,
            encode_chunk_size=10000,
            encode_batch_size=256,
        )

        for model_name, model_path in models.items():
            if all_results[dname].get(model_name):
                print(
                    f"Model {model_name} already evaluated for dataset {dname}. Skipping...",
                )
                continue

            print(f"Evaluating model: {model_name}")
            model = SentenceTransformer(model_path)
            all_results[dname][model_name] = evaluator(model)

        with open(f"{json_output_path}{dname}.json", "w", encoding="utf-8") as f:
            json.dump(all_results[dname], f, ensure_ascii=False, indent=4)

    # Plot the results for all datasets
    plot_beir_evaluation_results(
        json_output_path,
        output_dir_plots,
        metrics_to_plot=["mrr", "accuracy"],
    )
