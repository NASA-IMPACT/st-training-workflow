import argparse
import json
import os
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader
from custum_evals import (
    DummyModel,
    MultiGPUInformationRetrievalEvaluator,
    MultiGPUNanoBEIREvaluator,
    get_embedding_for_dataset,
)
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(description="Sentence Transformer Training Config")

parser.add_argument(
    "--dataset_name",
    type=str,
    default=None,
    choices=[
        "nanobeir",
        "beir",
        "nasa_sde_ir_v1",
        "nasa_sde_ir_v2",
        "nasa_sde_ir_v3",
        "nasa_smd_ir",
    ],
)
parser.add_argument("--ks", nargs="*", default=[1, 3, 5, 10])
parser.add_argument("--json_output_path", type=str, default="results_json/")
parser.add_argument("--output_dir_plots", type=str, default="results_plots/")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument(
    "--just_plot",
    type=int,
    default=0,
    choices=[0, 1],
    help="Set to 1 to just plot the results without running the evaluation.",
)
parser.add_argument(
    "--desired_metric_types",
    nargs="+",
    default=["mrr", "accuracy"],
    help="A list of metrics to plot (e.g., mrr, accuracy, ndcg, precision, recall, map).",
)


args = parser.parse_args()

dataset_name = args.dataset_name
ks = args.ks
json_output_path = args.json_output_path
output_dir_plots = args.output_dir_plots
batch_size = args.batch_size
just_plot = args.just_plot
desired_metric_types = args.desired_metric_types


os.makedirs(json_output_path, exist_ok=True)
os.makedirs(output_dir_plots, exist_ok=True)
json_output_path = f"{json_output_path}/{dataset_name}_eval_dump.json"


models = {
    "modernbert-embed-base": {
        "path": "nomic-ai/modernbert-embed-base",
        "color": "#1f77b4",
    },
    "nasa-smd-ibm-st-v2": {
        "path": "nasa-impact/nasa-smd-ibm-st-v2",
        "color": "#ff7f0e",
    },
    "indus-sde-st-v0.1": {"path": "nasa-impact/indus-sde-st-v0.1", "color": "#2ca02c"},
    "indus-sde-st-v0.2_whole-moon-14": {
        "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/"
        "model-qr3ln5om:v1/checkpoint-116000",
        "color": "#d62728",
    },
    "indus-sde-st-v0.2_atomic-plasma-15": {
        "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/"
        "model-ykf0bews:v1/checkpoint-13000",
        "color": "#9467bd",
    },
    "indus-sde-st-v0.2_vocal-river-16": {
        "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/"
        "model-dexwvlkj:v1/checkpoint-13000",
        "color": "#8c564b",
    },
    "indus-sde-st-v0.2_super-armadillo-25": {
        "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/"
        "model-1ogtkl75:v1/checkpoint-268500",
        "color": "#e377c2",
    },
    "indus-sde-st-v0.2_drawn-puddle-31": {
        "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/"
        "model-3x845j4d:v1/checkpoint-11000",
        "color": "#7f7f7f",
    },
    "indus-sde-st-v0.2_peach-night-57": {
        "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/"
        "model-rpv7vnpd:v1/checkpoint-13500",
        "color": "#bcbd22",
    },
    # "Qwen3-Embedding-0.6B": {
    #     "path": "Qwen/Qwen3-Embedding-0.6B",
    #     "color": "#bcbd22",
    #     "query_prompt": "Instruct: Given a search query (could be a question, title, or text), retrieve relevant scientific passages that answer or describe the query. \nQuery:",
    #     }
}

embeddings = {
    "[OpenAI]text-embedding-3-small": {
        "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/openai_emb_cache/"
        "text-embedding-3-small/",
        "model_name": "text-embedding-3-small",
        "color": "#17becf",
    },
    "[OpenAI]text-embedding-3-large": {
        "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/openai_emb_cache/"
        "text-embedding-3-large/",
        "model_name": "text-embedding-3-large",
        "color": "#393b79",
    },
}

dataset_config = {
    "nanobeir": {
        "path": None,
        "subsets": [None],  # this means to use all datasets
        "paths": {
            "NanoClimateFEVER": "zeta-alpha-ai/NanoClimateFEVER",
            "NanoDBPedia": "zeta-alpha-ai/NanoDBPedia",
            "NanoFEVER": "zeta-alpha-ai/NanoFEVER",
            "NanoFiQA2018": "zeta-alpha-ai/NanoFiQA2018",
            "NanoHotpotQA": "zeta-alpha-ai/NanoHotpotQA",
            "NanoMSMARCO": "zeta-alpha-ai/NanoMSMARCO",
            "NanoNFCorpus": "zeta-alpha-ai/NanoNFCorpus",
            "NanoNQ": "zeta-alpha-ai/NanoNQ",
            "NanoQuoraRetrieval": "zeta-alpha-ai/NanoQuoraRetrieval",
            "NanoSCIDOCS": "zeta-alpha-ai/NanoSCIDOCS",  # issue
            "NanoArguAna": "zeta-alpha-ai/NanoArguAna",
            "NanoSciFact": "zeta-alpha-ai/NanoSciFact",
            "NanoTouche2020": "zeta-alpha-ai/NanoTouche2020",
        },
    },
    "beir": {
        "path": None,
        "subsets": [
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
        ],
        "dataset_cache_path": "./beir_datasets",
    },
    "nasa_sde_ir_v1": {"path": "nasa-impact/nasa-sde-IR-benchmark-sample-v1"},
    "nasa_sde_ir_v2": {"path": "nasa-impact/nasa-sde-IR-benchmark-sample-v2"},
    "nasa_smd_ir": {"path": "nasa-impact/nasa-smd-IR-benchmark"},
    "nasa_sde_ir_v3": {
        "path": "nasa-impact/nasa-sde-IR-benchmark-sample-v3",
        "data_files": [
            "qrels/question-answer~SDE_general_v2.tsv",
            "qrels/question-answer~SDE_general_v3.tsv",
            "qrels/search_term-document~CMR.tsv",
            "qrels/search_term-document~PDS.tsv",
            "qrels/search_term-document~SDE_general_v2.tsv",
            "qrels/search_term-document~SDE_general_v3.tsv",
            "qrels/title-description~CMR.tsv",
            "qrels/title-description~PDS.tsv",
        ],
    },
}


def dataset_getter(
    dataset_name,
    corpus_split="train",
    queries_split="train",
    relevant_docs_split="test",
    data_file=None,
):
    corpus = load_dataset(
        dataset_config[dataset_name]["path"],
        data_files="corpus.jsonl",
        split=corpus_split,
    )
    queries = load_dataset(
        dataset_config[dataset_name]["path"],
        data_files="queries.jsonl",
        split=queries_split,
    )
    relevant_docs_data = load_dataset(
        dataset_config[dataset_name]["path"],
        split=relevant_docs_split,
        data_files=data_file,
    )

    corpus = {row["_id"]: row["text"] for i, row in enumerate(corpus)}
    queries = {row["_id"]: row["text"] for row in queries}
    relevant_docs_data = (
        relevant_docs_data.to_pandas()
        .groupby("query-id")["corpus-id"]
        .apply(set)
        .to_dict()
    )
    relevant_docs_data = {
        str(k): {str(item) for item in v} for k, v in relevant_docs_data.items()
    }

    return corpus, queries, relevant_docs_data


def beir_dataset_getter(subset):
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{subset}.zip"
    out_dir = os.path.join(
        pathlib.Path.cwd(),
        dataset_config["beir"]["dataset_cache_path"],
    )
    data_path = beir_util.download_and_unzip(url, out_dir)

    # Load the dataset using the BEIR loader
    corpus_beir, queries, qrels_beir = GenericDataLoader(data_folder=data_path).load(
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

    return corpus, queries, relevant_docs


def get_dataset(dataset_name, subset=None, relevant_docs_split="test", data_file=None):
    if dataset_name.lower() in ["beir"]:
        return beir_dataset_getter(subset)
    elif dataset_name.lower() in ["nanobeir"]:
        return None, None, None
    else:
        return dataset_getter(
            dataset_name,
            relevant_docs_split=relevant_docs_split,
            data_file=data_file,
        )


def get_evaluator(
    dataset_name: str,
    queries: dict,
    corpus: dict,
    relevant_docs_data: dict,
    subset=None,
    data_file=None,
):
    if dataset_name.lower() == "nanobeir":
        evaluator = MultiGPUNanoBEIREvaluator(
            dataset_names=None,
            mrr_at_k=ks,
            accuracy_at_k=ks,
            precision_recall_at_k=ks,
            map_at_k=ks,
            ndcg_at_k=ks,
            show_progress_bar=True,
            batch_size=batch_size,
            write_csv=True,
        )

    elif dataset_name.lower() == "beir":
        evaluator = MultiGPUInformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs_data,
            name=f"beir__{subset}____evaluator",
            batch_size=batch_size,
            mrr_at_k=ks,
            ndcg_at_k=ks,
            accuracy_at_k=ks,
            precision_recall_at_k=ks,
            map_at_k=ks,
            show_progress_bar=True,
            write_csv=True,
            encode_chunk_size=5000,
            encode_batch_size=batch_size,
        )

    elif dataset_name.lower() == "nasa_sde_ir_v1":
        evaluator = MultiGPUInformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs_data,
            name=f"{dataset_name}______evaluator",
            batch_size=batch_size,
            mrr_at_k=ks,
            ndcg_at_k=ks,
            accuracy_at_k=ks,
            precision_recall_at_k=ks,
            map_at_k=ks,
            show_progress_bar=True,
            write_csv=True,
            encode_chunk_size=5000,
            encode_batch_size=batch_size,
        )

    elif dataset_name.lower() == "nasa_sde_ir_v2":
        evaluator = MultiGPUInformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs_data,
            name=f"{dataset_name}______evaluator",
            batch_size=batch_size,
            mrr_at_k=ks,
            ndcg_at_k=ks,
            accuracy_at_k=ks,
            precision_recall_at_k=ks,
            map_at_k=ks,
            show_progress_bar=True,
            write_csv=True,
            encode_chunk_size=5000,
            encode_batch_size=batch_size,
        )

    elif dataset_name.lower() == "nasa_sde_ir_v3":
        evaluator = MultiGPUInformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs_data,
            name=f"{dataset_name}____{data_file}__evaluator",
            batch_size=batch_size,
            mrr_at_k=ks,
            ndcg_at_k=ks,
            accuracy_at_k=ks,
            precision_recall_at_k=ks,
            map_at_k=ks,
            show_progress_bar=True,
            write_csv=True,
            encode_chunk_size=5000,
            encode_batch_size=batch_size,
        )

    elif dataset_name.lower() == "nasa_smd_ir":
        evaluator = MultiGPUInformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs_data,
            name=f"{dataset_name}______evaluator",
            batch_size=batch_size,
            mrr_at_k=ks,
            ndcg_at_k=ks,
            accuracy_at_k=ks,
            precision_recall_at_k=ks,
            map_at_k=ks,
            show_progress_bar=True,
            write_csv=True,
            encode_chunk_size=5000,
            encode_batch_size=batch_size,
        )

    return evaluator


def add_mean_metrics(all_results):
    for model_name in all_results:
        metric_names = set(
            [i.split("_")[-1] for i in list(all_results[model_name].keys())],
        )
        subset_names = set(
            [
                i.split("__")[1]
                for i in all_results[model_name]
                if "mean" not in i.split("__")[1]
            ],
        )

        mean_result = {}

        # Calculate the mean for each metric
        for metric in metric_names:
            values = []
            for subset in subset_names:
                key_name = f"{dataset_name}__{subset}____evaluator_cosine_{metric}"
                values.append(all_results[model_name][key_name])

            mean_result[f"{dataset_name}__mean____evaluator_cosine_{metric}"] = sum(
                values,
            ) / (len(values) if len(values) > 0 else 1)

        all_results[model_name] = {**all_results[model_name], **mean_result}


def check_if_eval_already_exists(all_results, model_name, subset=None, data_file=None):
    if model_name not in all_results:
        return False

    # get set of all data_files for the model_name
    existing_data_files = set(
        [k.split("__")[2] for k in all_results.get(model_name, {}).keys()],
    )
    if data_file is not None and data_file not in existing_data_files:
        return False

    # similarly get all the subsets for the model_name
    existing_subsets = set(
        [k.split("__")[1] for k in all_results.get(model_name, {}).keys()],
    )
    if subset is not None and subset not in existing_subsets:
        return False

    return True


def pre_compute_corpus_embedding(
    models,
    dataset_name,
    subset,
    all_results,
    dataset_config,
):
    data_file = dataset_config[dataset_name].get("data_files", [None])[0]
    n_data_files = len(data_file)
    corpus, _q, _ = get_dataset(
        dataset_name,
        subset,
        relevant_docs_split="train" if n_data_files > 1 else "test",
        data_file=None,
    )

    corpus_texts = list(corpus.values())
    corpus_pre_computed_embeddings = {}
    for model_name, model_info in models.items():

        if check_if_eval_already_exists(all_results, model_name, subset, data_file):
            print(
                f"Model {model_name} with the subset {subset} and data_file {data_file} already evaluated. Skipping...Preembedding of corpus",
            )
            continue

        print(
            f"Pre-computing corpus embeddings for model: {model_name}, subset: {subset}",
        )
        model = SentenceTransformer(model_info["path"])

        pool = model.start_multi_process_pool()

        corpus_embeddings = model.encode(
            corpus_texts,
            pool=pool,
            batch_size=batch_size,
            chunk_size=5000,
            show_progress_bar=True,
        )

        model.stop_multi_process_pool(pool)
        corpus_pre_computed_embeddings[
            f"{dataset_name}__{subset}__{model_name}"
        ] = corpus_embeddings

    return corpus_pre_computed_embeddings


def evaluate():
    # check if the json_output_path file exists
    # if it does, load it to all_results else initilize an empty dictionary
    if os.path.exists(json_output_path):
        # load the json
        with open(json_output_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    subsets = dataset_config[dataset_name].get("subsets", [None])
    for subset in subsets:
        # this will loop multiple times if subsets are provided else it will loop once
        # check if there is multiple data_files for the dataset_name

        # precompute corpus embeddings for different dataset_name-subset-model_name
        ## for different data_files, only relevant_docs / qrels are different
        corpus_pre_computed_embeddings = pre_compute_corpus_embedding(
            models,
            dataset_name,
            subset,
            all_results,
            dataset_config,
        )
        for data_file in dataset_config[dataset_name].get("data_files", [None]):
            corpus, queries, relevant_docs = get_dataset(
                dataset_name,
                subset,
                relevant_docs_split="test" if data_file is None else "train",
                data_file=data_file,
            )
            evaluator = get_evaluator(
                dataset_name,
                queries,
                corpus,
                relevant_docs,
                subset,
                data_file,
            )
            # Looping models
            for model_name, model_info in models.items():
                print(
                    f"Evaluating model: {model_name}, subset {subset} and data_file: {data_file}",
                )
                if check_if_eval_already_exists(
                    all_results,
                    model_name,
                    subset,
                    data_file,
                ):
                    print(
                        f"Model {model_name} with the subset {subset} and data_file {data_file} already evaluated. Skipping...",
                    )
                    continue
                model = SentenceTransformer(model_info["path"])
                results = evaluator(
                    model,
                    query_prompt_str=model_info.get("query_prompt", None),
                    corpus_embeddings=corpus_pre_computed_embeddings.get(
                        f"{dataset_name}__{subset}__{model_name}",
                    ),
                )
                results = {
                    k: v for k, v in results.items() if k.startswith(dataset_name)
                }  # filtering out non compatible keys
                if model_name not in all_results:
                    all_results[model_name] = {}
                all_results[model_name] = {**all_results[model_name], **results}

            # Looping through the embeddings
            for embedding_name, embedding_info in embeddings.items():
                print(
                    f"Evaluating embedding: {embedding_name}, subset {subset} and data_file: {data_file}",
                )
                if check_if_eval_already_exists(
                    all_results,
                    embedding_name,
                    subset,
                    data_file,
                ):
                    print(
                        f"Embedding {embedding_name} already evaluated in json. Skipping...",
                    )
                    continue

                print(
                    f"Loading embeddings for {embedding_name} from {embedding_info['path']}",
                )
                # check if the embedding for the dataset_name exists if not generate it: calling a function
                corpus_df, queries_df = get_embedding_for_dataset(
                    dataset_config=dataset_config[dataset_name],
                    embedding_path=embedding_info["path"],
                    dataset_name=dataset_name,
                    model_name=embedding_info["model_name"],
                    subset=subset,
                    data_file=data_file,
                )

                dummy_model = DummyModel()

                if isinstance(corpus_df, pd.DataFrame) and isinstance(
                    queries_df,
                    pd.DataFrame,
                ):
                    results = evaluator(
                        model=dummy_model,
                        corpus_df=corpus_df,
                        query_df=queries_df,
                    )
                elif isinstance(corpus_df, dict) and isinstance(queries_df, dict):
                    results = evaluator(
                        model=dummy_model,
                        corpus_dfs=corpus_df,
                        query_dfs=queries_df,
                    )
                    # filteriing the results to only include the valid keys
                    results = {
                        k: v for k, v in results.items() if k.startswith(dataset_name)
                    }

                if embedding_name not in all_results:
                    all_results[embedding_name] = {}
                all_results[embedding_name] = {**all_results[embedding_name], **results}

    if len(subsets) > 1 or len(dataset_config[dataset_name].get("paths", [])) > 1:
        # need to add a mean of metrics from different subsets of different models
        add_mean_metrics(all_results)

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)


def plot_results(json_output_path):
    print("Plotting results...")
    if not os.path.exists(json_output_path):
        print(
            f"JSON output path {json_output_path} does not exist. Please run the evaluation first.",
        )
        return
    with open(json_output_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    # lets make a df first of the json
    records = []
    for model, metrics in all_results.items():
        for metric_name_full, value in metrics.items():
            parts = metric_name_full.split("__")
            dataset_name_str = parts[0]
            subset_str = parts[1]
            data_file_str = parts[2]
            remaining_str = parts[-1]

            parts = remaining_str.split("@")
            k = int(parts[1])
            metric_name = parts[0].split("_")[-1]
            records.append(
                {
                    "dataset_name": dataset_name_str,
                    "subset": subset_str,
                    "data_file": data_file_str,
                    "model": model,
                    "metric": metric_name,
                    "k": k,
                    "value": value,
                },
            )

    # Create a pandas DataFrame
    df = pd.DataFrame(records)

    # filter the DataFrame to include only the desired metric types
    df = df[df["metric"].isin(desired_metric_types)]

    all_model_configs = {**models, **embeddings}
    model_color_palette = {
        name: config["color"] for name, config in all_model_configs.items()
    }
    sorted_model_names = sorted(all_model_configs.keys())
    # lets loop through subsets as we will be plotting them separately
    for subset in df["subset"].unique():
        subset_df = df[df["subset"] == subset]

        for data_file in subset_df["data_file"].unique():
            subset_data_file_df = subset_df[subset_df["data_file"] == data_file]

            # Create the bar plot using seaborn's catplot for faceting
            # Create the bar plot
            g = sns.catplot(
                data=subset_data_file_df,
                x="k",
                y="value",
                hue="model",
                hue_order=sorted_model_names,
                col="metric",
                kind="bar",
                col_wrap=2,
                sharey=False,
                height=5,
                aspect=2,
                legend_out=True,
                palette=model_color_palette,
            )

            # Customize subplot titles and labels
            g.set_titles("Metric: {col_name}")
            g.set_axis_labels("K Value", "Score")
            g.despine(left=True)

            # Add value labels on top of each bar
            for ax in g.axes.flat:
                for p in ax.patches:
                    value = f"{p.get_height():.2f}"
                    x = p.get_x() + p.get_width() / 2
                    y = p.get_height()
                    ax.annotate(
                        value,
                        (x, y),
                        ha="center",
                        va="center",
                        xytext=(0, 5),
                        textcoords="offset points",
                        fontsize=9,
                    )

            # 1. Move the legend to be centered below the plot
            sns.move_legend(
                g,
                "lower center",
                bbox_to_anchor=(
                    0.5,
                    -0.2,
                ),  # Center the legend horizontally, move it down
                ncol=5,  # Adjust number of columns to fit your models (image has 10)
                title=None,
                frameon=False,
            )

            # 2. Add the main title for the figure
            title = f"Model Performance at K-Value On {dataset_name}"
            if subset != "":
                title += f" - Subset: {subset}"
            if data_file is not None or data_file != "":
                title += f" - Data File: {data_file}"
            g.fig.suptitle(
                title,
                fontsize=16,  # Optional: Adjust font size
            )

            # 3. Use tight_layout to automatically adjust spacing and center the title
            # The rect parameter makes space for the suptitle at the top
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            # 4. Save the figure
            # The bbox_inches="tight" argument is crucial for including the legend
            os.makedirs(os.path.join(output_dir_plots, dataset_name), exist_ok=True)
            plt.savefig(
                os.path.join(
                    output_dir_plots,
                    dataset_name,
                    f"{dataset_name}_{subset}_{data_file.split('/')[-1]}_performance_plots.png",
                ),
                bbox_inches="tight",
                dpi=300,  # Optional: Increase image resolution
            )


if __name__ == "__main__":
    if not just_plot:
        evaluate()
    plot_results(json_output_path)
