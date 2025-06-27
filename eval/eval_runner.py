import json

import sentence_transformers.util as util
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

# Define the device to use
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
ks = [1, 3, 5, 10]

models = {
    "modernbert-embed-base": "nomic-ai/modernbert-embed-base",
    "nasa-smd-ibm-st-v2": "nasa-impact/nasa-smd-ibm-st-v2",
    "indus-sde-st-v0.1": "nasa-impact/indus-sde-st-v0.1",
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
)

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

with open("nasa_ir_eval_dump.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)
