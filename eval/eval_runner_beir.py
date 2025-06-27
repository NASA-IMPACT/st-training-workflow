import json
import logging
import os
import pathlib
from collections import defaultdict

import torch._dynamo
from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.evaluation import InformationRetrievalEvaluator

torch._dynamo.config.suppress_errors = True


#### Just some setup for logging ####
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


os.makedirs("beir_eval_results", exist_ok=True)

all_results = {}

split = "test"
ks = [1, 3, 5, 10]

models = {
    "modernbert-embed-base": "nomic-ai/modernbert-embed-base",
    "nasa-smd-ibm-st-v2": "nasa-impact/nasa-smd-ibm-st-v2",
    "indus-sde-st-v0.1": "nasa-impact/indus-sde-st-v0.1",
}

_DATASETS = [
    # "trec-covid",
    #  "nfcorpus",
    #  "nq", "hotpotqa", "fiqa",
    #  "arguana", "webis-touche2020", "dbpedia-entity","scidocs", "fever",
    #  "climate-fever", "scifact",
    "msmarco",
    "quora",
    "cqadupstack",
]


# start to evaluate
for dname in _DATASETS:
    if all_results.get(dname) is None:
        all_results[dname] = {}

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dname}.zip"
    out_dir = os.path.join(pathlib.Path.cwd(), "datasets")
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

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=f"{dname}_evaluator",
        batch_size=8,
        mrr_at_k=ks,
        ndcg_at_k=ks,
        accuracy_at_k=ks,
        precision_recall_at_k=ks,
        map_at_k=ks,
        show_progress_bar=True,
        write_csv=True,
    )

    for model_name, model_path in models.items():
        print(f"Evaluating model: {model_name}")
        model = SentenceTransformer(model_path)
        all_results[dname][model_name] = evaluator(model)

        # Clear model from memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    with open(f"./beir_eval_results/{dname}.json", "w", encoding="utf-8") as f:
        json.dump(all_results[dname], f, ensure_ascii=False, indent=4)


# with open("./beir_eval_results/BEIR_eval_dump.json", "w", encoding="utf-8") as f:
#     json.dump(all_results, f, ensure_ascii=False, indent=4)
