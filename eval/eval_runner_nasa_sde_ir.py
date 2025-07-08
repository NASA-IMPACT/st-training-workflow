import json
import os

import sentence_transformers.util as util
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

# Define the device to use
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
ks = [1, 3, 5, 10]
json_output_path = "nasa_sde_ir_eval_dump.json"

models = {
    "modernbert-embed-base": "nomic-ai/modernbert-embed-base",
    "nasa-smd-ibm-st-v2": "nasa-impact/nasa-smd-ibm-st-v2",
    "indus-sde-st-v0.1": "nasa-impact/indus-sde-st-v0.1",
    "indus-sde-st-v0.2_whole-moon-14": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/model-qr3ln5om:v1/checkpoint-116000",
}

corpus = load_dataset(
    "nasa-impact/nasa-sde-IR-benchmark",
    data_files="corpus.jsonl",
    split="train",
)
queries = load_dataset(
    "nasa-impact/nasa-sde-IR-benchmark",
    data_files="queries.jsonl",
    split="train",
)
relevant_docs_data = load_dataset("nasa-impact/nasa-sde-IR-benchmark", split="test")


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
    name="nasa_sde_ir_evaluator",
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
