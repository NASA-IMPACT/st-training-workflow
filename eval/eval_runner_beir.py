import json
import logging
import os
import pathlib
from collections import defaultdict

import torch._dynamo
from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader
from custum_evals import MultiGPUInformationRetrievalEvaluator
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

os.makedirs(json_output_path, exist_ok=True)


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
    "msmarco",
    "quora",
    # "cqadupstack",
]

all_results = {}
# for dname in _DATASETS:
#     if os.path.exists(f"{json_output_path}{dname}.json"):
#         # load the json
#         with open(f"{json_output_path}{dname}.json", "r", encoding="utf-8") as f:
#             all_results[dname] = json.load(f)
#     else:
#         all_results[dname] = {}


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

    # with open("./beir_eval_results/BEIR_eval_dump.json", "w", encoding="utf-8") as f:
    #     json.dump(all_results, f, ensure_ascii=False, indent=4)
