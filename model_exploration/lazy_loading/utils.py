import os
import time
import argparse
import datetime
import random
from typing import List, Dict, Optional, Union
from datasets import (
    load_dataset,
    load_from_disk,
    Dataset,
    DatasetDict,
    concatenate_datasets,
    get_dataset_config_names,
    Features, Value
)
from datasets import IterableDataset, interleave_datasets, DatasetDict as HFDatasetDict, Dataset as HFDataset

import csv # For CSV writing
import wandb
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, InputExample
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments, BatchSamplers
from sentence_transformers.evaluation import InformationRetrievalEvaluator, TripletEvaluator, SequentialEvaluator, SentenceEvaluator
from dotenv import load_dotenv
import distributed
from distributed import init_ddp, print0
from torch.nn.parallel import DistributedDataParallel


class TimedIREvaluator(InformationRetrievalEvaluator):
    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        start = time.perf_counter()
        results = super().__call__(model, output_path, epoch, steps)
        elapsed = time.perf_counter() - start

        n_q = len(self.queries)
        avg = elapsed / n_q
        print(f"[TimedIREvaluator] total {elapsed:.2f}s over {n_q} queries → avg {avg*1000:.1f} ms/query")

        # optionally add to the results dict for logging to CSV
        results["retrieval_time_total_s"] = elapsed
        results["retrieval_time_avg_s"]   = avg
        return results

def get_gpu_info():
    """
    Returns a dict with:
      - gpu_available: bool
      - gpu_count:     int
      - gpu_names:     list of strings
    """
    gpu_available = torch.cuda.is_available()
    if not gpu_available:
        return {"gpu_available": False, "gpu_count": 0, "gpu_names": []}

    gpu_count = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    return {
        "gpu_available": True,
        "gpu_count": gpu_count,
        "gpu_names": gpu_names
    }


# Keep your original split_dataset for non-streaming cases
def original_split_dataset(
    ds: HFDataset, # Expects a regular Hugging Face Dataset
    train_split_name: str = "train",
    val_split_name:   str = "validation",
    test_split_name:  str = "test",
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 42
) -> HFDatasetDict:
    full = ds
    combined = val_frac + test_frac

    if combined == 0: # No validation or test set
        return HFDatasetDict({
            train_split_name: full,
            val_split_name:   HFDataset.from_dict({col: [] for col in full.features.keys()}, features=full.features),
            test_split_name:  HFDataset.from_dict({col: [] for col in full.features.keys()}, features=full.features),
        })
    if combined >= 1.0:
        raise ValueError("val_frac + test_frac must be less than 1.0 if both are non-zero")

    split1 = full.train_test_split(test_size=combined, seed=seed, shuffle=True)
    train_ds, eval_ds = split1["train"], split1["test"]

    if val_frac == 0: # Only test split from eval_ds
        val_ds = HFDataset.from_dict({col: [] for col in eval_ds.features.keys()}, features=eval_ds.features)
        test_ds = eval_ds
    elif test_frac == 0: # Only val split from eval_ds
        val_ds = eval_ds
        test_ds = HFDataset.from_dict({col: [] for col in eval_ds.features.keys()}, features=eval_ds.features)
    else: # Both val and test
        val_rel = val_frac / combined
        split2 = eval_ds.train_test_split(test_size=(1.0 - val_rel), seed=seed, shuffle=True) # test_size is for the second part (test)
        val_ds, test_ds = split2["train"], split2["test"]
        
    return HFDatasetDict({
        train_split_name: train_ds,
        val_split_name:   val_ds,
        test_split_name:  test_ds,
    })


def split_iterable_dataset(
    iterable_ds: IterableDataset,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    total_samples_if_known: Optional[int] = None,
    seed: int = 42,
    shuffle_buffer_size: int = 10000,
    name: str = "streamed_dataset" # For logging
) -> Dict[str, IterableDataset]:
    """
    Splits an IterableDataset into train, validation, and test iterable splits.
    If total_samples_if_known, splits by fraction. Otherwise, uses fixed N for val/test.
    Important: This creates new iterators that consume the original when iterated.
    """
    print(f"Splitting IterableDataset: {name}")
    shuffled_ds = iterable_ds.shuffle(seed=seed, buffer_size=shuffle_buffer_size)

    num_val_samples: int
    num_test_samples: int

    if total_samples_if_known and total_samples_if_known > 0 :
        if val_frac + test_frac >= 1.0:
             raise ValueError(f"For {name}, val_frac ({val_frac}) + test_frac ({test_frac}) must be < 1.0")
        num_val_samples = int(total_samples_if_known * val_frac)
        num_test_samples = int(total_samples_if_known * test_frac)
        # Train gets the rest
    else:
        if total_samples_if_known == 0: # Handle case where NROWS might be 0
            num_val_samples = 0
            num_test_samples = 0
        else:
            print(f"Warning: total_samples_if_known is None or 0 for streaming split of {name}. Using fixed N (max 1000) for val/test if fractions > 0.")
            num_val_samples = min(1000, int(0.05 * 20000)) if val_frac > 0 else 0 # Default to 5% of 20k or 1k
            num_test_samples = min(1000, int(0.05 * 20000)) if test_frac > 0 else 0 # Default to 5% of 20k or 1k
    
    print(f"IterableDataset split for {name}: val_samples={num_val_samples}, test_samples={num_test_samples}")


    val_ds = shuffled_ds.take(num_val_samples)
    test_ds = shuffled_ds.skip(num_val_samples).take(num_test_samples)
    train_ds = shuffled_ds.skip(num_val_samples + num_test_samples) # Takes the rest

    return {
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    }



def get_all_data_subset(name: str, path: str, s1: str, s2: str, loss_fn) -> dict:
    """
    Auto-generate configs for all dataset variants under `path`.
    """
    out = {}
    for cfg in get_dataset_config_names(path):
        key = f"{name}_{cfg}"
        out[key] = {
            "args":   {"path": path, "name": cfg},
            "map_fn": lambda ex, s1=s1, s2=s2: {"anchor": ex[s1], "positive": ex[s2]},
            "cols": Features({"anchor": Value("string"),"positive": Value("string")}),
            "loss":   loss_fn
        }
    return out


def build_dataset_configs(N_DATA_SRC=None) -> dict:
    """
    Define all your dataset mappings and losses.
    """
    base = {
        "squad_v2": {
        "args": {"path": "rajpurkar/squad_v2"},
        "map_fn": lambda ex: {"anchor": ex["question"], "positive": ex["context"]},
        "cols": Features({"anchor": Value("string"),"positive": Value("string")}),
        "loss": MultipleNegativesRankingLoss
        },
        "wikipedia": {
            "args": {"path": "wikimedia/wikipedia", "data_dir": "20231101.en"},
            "map_fn": lambda ex: {"anchor": ex["title"], "positive": ex["text"]},
            "cols": Features({"anchor": Value("string"),"positive": Value("string")}),
            "loss": MultipleNegativesRankingLoss
        },
        "StackExchange_Math_titlebody_answer": {
            "args": {"path": "flax-sentence-embeddings/stackexchange_math_jsonl", "data_dir": "titlebody_answer"},
            "map_fn": lambda ex: {"anchor": ex["title"], "positive": ex["upvoted_answer"]},
            "cols": Features({"anchor": Value("string"),"positive": Value("string")}),
            "loss": MultipleNegativesRankingLoss
        },
        "StackExchange_Math_title_answer": {
            "args": {"path": "flax-sentence-embeddings/stackexchange_math_jsonl", "data_dir": "title_answer"},
            "map_fn": lambda ex: {"anchor": ex["title"], "positive": ex["upvoted_answer"]},
            "cols": Features({"anchor": Value("string"),"positive": Value("string")}),
            "loss": MultipleNegativesRankingLoss
        },
        # "StackExchange_title_body": {
        #     "args": {"path": "flax-sentence-embeddings/stackexchange_title_body_jsonl"},
        #     "map_fn": lambda ex: {"anchor": ex["texts"][0], "positive": ex["texts"][1]},
        #     "loss": MultipleNegativesRankingLoss
        # },
        "StackExchange_title_body": {
            "args": {"path": "flax-sentence-embeddings/stackexchange_title_body_jsonl"},
            "map_fn": lambda batch: {
                "anchor": [texts_pair[0] for texts_pair in batch["texts"] if len(texts_pair) >= 2], # Added safety check
                "positive": [texts_pair[1] for texts_pair in batch["texts"] if len(texts_pair) >= 2] # Added safety check
            },
            "cols": Features({"anchor": Value("string"),"positive": Value("string")}),
            "loss": MultipleNegativesRankingLoss
        },
        "StackExchange_Duplicates_titlebody_titlebody": {
            "args": {"path": "sentence-transformers/stackexchange-duplicates", "data_dir": "post-post-pair"},
            "map_fn": lambda ex: {"anchor": ex["post1"], "positive": ex["post2"]},
            "cols": Features({"anchor": Value("string"),"positive": Value("string")}),
            "loss": MultipleNegativesRankingLoss
        },
        "StackExchange_Duplicates_body_body": {
            "args": {"path": "sentence-transformers/stackexchange-duplicates", "data_dir": "body-body-pair"},
            "map_fn": lambda ex: {"anchor": ex["body1"], "positive": ex["body2"]},
            "cols": Features({"anchor": Value("string"),"positive": Value("string")}),
            "loss": MultipleNegativesRankingLoss
        },
        "StackExchange_Duplicates_title_title": {
            "args": {"path": "sentence-transformers/stackexchange-duplicates", "data_dir": "title-title-pair"},
            "map_fn": lambda ex: {"anchor": ex["title1"], "positive": ex["title2"]},
            "cols": Features({"anchor": Value("string"),"positive": Value("string")}),
            "loss": MultipleNegativesRankingLoss
        },
        # "WikiAnswer_Pairs": {
        #     "args": {"path": "sentence-transformers/wikianswers-duplicates"},
        #     "map_fn": lambda ex: {"anchor": ex["anchor"], "positive": ex["positive"]},
        #     "loss": MultipleNegativesRankingLoss
        # },
        "Natural_Questions": {
            "args": {"path": "sentence-transformers/natural-questions"},
            "map_fn": lambda ex: {"anchor": ex["query"], "positive": ex["answer"]},
            "cols": Features({"anchor": Value("string"),"positive": Value("string")}),
            "loss": MultipleNegativesRankingLoss
        },
        # "PAQ": {
        #     "args": {"path": "embedding-data/PAQ_pairs"},
        #     "map_fn": lambda ex: {"anchor": ex["set"][0], "positive": ex["set"][1]},
        #     "loss": MultipleNegativesRankingLoss
        # },
        "PAQ": {
            "args": {"path": "embedding-data/PAQ_pairs"},
            "map_fn": lambda batch: {
                "anchor": [text_set[0] for text_set in batch["set"] if len(text_set) >= 2], # Added safety check
                "positive": [text_set[1] for text_set in batch["set"] if len(text_set) >= 2] # Added safety check
            },
            "cols": Features({"anchor": Value("string"),"positive": Value("string")}),
            "loss": MultipleNegativesRankingLoss
        },
        "Gooaq": {
            "args": {"path": "sentence-transformers/gooaq"},
            "map_fn": lambda ex: {"anchor": ex["question"], "positive": ex["answer"]},
            "cols": Features({"anchor": Value("string"),"positive": Value("string")}),
            "loss": MultipleNegativesRankingLoss
        },
        "yahoo_title_answers": {
            "args": {"path": "sentence-transformers/yahoo-answers", "data_dir": "title-answer-pair"},
            "map_fn": lambda ex: {"anchor": ex["title"], "positive": ex["answer"]},
            "cols": Features({"anchor": Value("string"),"positive": Value("string")}),
            "loss": MultipleNegativesRankingLoss
        },
        "msmacro_triplet": {
            "args": {"path": "sentence-transformers/msmarco-msmarco-MiniLM-L6-v3", "data_dir": "triplet-hard"},
            "map_fn": lambda ex: {"anchor": ex["query"], "positive": ex["positive"], "negative": ex["negative"]},
            "cols": Features({"anchor": Value("string"),"positive": Value("string"), "negative": Value("string")}),
            "loss": MultipleNegativesRankingLoss
        },
        "trivia_qa_triplet": {
            "args": {"path": "sentence-transformers/trivia-qa-triplet", "data_dir": "triplet-all"},
            "map_fn": lambda ex: {"anchor": ex["anchor"], "positive": ex["positive"], "negative": ex["negative"]},
            "cols": Features({"anchor": Value("string"),"positive": Value("string"), "negative": Value("string")}),
            "loss": MultipleNegativesRankingLoss
        },
        "nli_for_simcse_triplet": {
            "args": {"path": "sentence-transformers/nli-for-simcse", "data_dir": "triplet-all"},
            "map_fn": lambda ex: {"anchor": ex["anchor"], "positive": ex["positive"], "negative": ex["negative"]},
            "cols": Features({"anchor": Value("string"),"positive": Value("string"), "negative": Value("string")}),
            "loss": MultipleNegativesRankingLoss
        },
        "quora_dup_triplet": {
            "args": {"path": "sentence-transformers/quora-duplicates", "data_dir": "triplet-all"},
            "map_fn": lambda ex: {"anchor": ex["anchor"], "positive": ex["positive"], "negative": ex["negative"]},
            "cols": Features({"anchor": Value("string"),"positive": Value("string"), "negative": Value("string")}),
            "loss": MultipleNegativesRankingLoss
        },
        # "WikiAnswers": {
        #     "args": {"path": "embedding-data/WikiAnswers"},
        #     "map_fn": lambda ex: dict(zip(("anchor", "positive"), random.sample(ex["set"], 2))),
        #     "loss": MultipleNegativesRankingLoss
        # },
        "WikiAnswers": {
            "args": {"path": "embedding-data/WikiAnswers"},
            "map_fn": lambda batch: {
                # This creates a list of dictionaries, then transposes it
                k: [dic[k] for dic in (
                        dict(zip(("anchor", "positive"), random.sample(s, 2))) if len(s) >= 2
                        else {"anchor": s[0] if len(s) == 1 else None, "positive": s[0] if len(s) == 1 else None} # Handle sets with < 2 items
                     for s in batch["set"])]
                for k in ("anchor", "positive")
            },
            "cols": Features({"anchor": Value("string"),"positive": Value("string")}),
            "loss": MultipleNegativesRankingLoss
        },
        "eli5": {
            "args": {"path": "sentence-transformers/eli5"},
            "map_fn": lambda ex: {"anchor": ex["question"], "positive": ex["answer"]},
            "cols": Features({"anchor": Value("string"),"positive": Value("string")}),
            "loss": MultipleNegativesRankingLoss
        },
        "sentence_compression": {
            "args": {"path": "sentence-transformers/sentence-compression"},
            "map_fn": lambda ex: {"anchor": ex["simplified"], "positive": ex["text"]},
            "cols": Features({"anchor": Value("string"),"positive": Value("string")}),
            "loss": MultipleNegativesRankingLoss
        },
        "Flickr30k_Captions": {
            "args": {"path": "sentence-transformers/flickr30k-captions"},
            "map_fn": lambda ex: {"anchor": ex["caption1"], "positive": ex["caption2"]},
            "cols": Features({"anchor": Value("string"),"positive": Value("string")}),
            "loss": MultipleNegativesRankingLoss
        },
        "Coco_Captions": {
            "args": {"path": "sentence-transformers/coco-captions"},
            "map_fn": lambda ex: {"anchor": ex["caption1"], "positive": ex["caption2"]},
            "cols": Features({"anchor": Value("string"),"positive": Value("string")}),
            "loss": MultipleNegativesRankingLoss
        },
        "xsum": {
            "args": {"path": "sentence-transformers/xsum"},
            "map_fn": lambda ex: {"anchor": ex["article"], "positive": ex["summary"]},
            "cols": Features({"anchor": Value("string"),"positive": Value("string")}),
            "loss": MultipleNegativesRankingLoss
        },
        "agnews": {
            "args": {"path": "sentence-transformers/agnews"},
            "map_fn": lambda ex: {"anchor": ex["title"], "positive": ex["description"]},
            "cols": Features({"anchor": Value("string"),"positive": Value("string")}),
            "loss": MultipleNegativesRankingLoss
        },
        "npr": {
            "args": {"path": "sentence-transformers/npr"},
            "map_fn": lambda ex: {"anchor": ex["title"], "positive": ex["body"]},
            "cols": Features({"anchor": Value("string"),"positive": Value("string")}),
            "loss": MultipleNegativesRankingLoss
        },
        "cnn_dailymail": {
            "args": {"path": "abisee/cnn_dailymail", "name": "3.0.0"},
            "map_fn": lambda ex: {"anchor": ex["highlights"], "positive": ex["article"]},
            "cols": Features({"anchor": Value("string"),"positive": Value("string")}),
            "loss": MultipleNegativesRankingLoss
        },
        "cc_news": {
            "args": {"path": "vblagoje/cc_news"},
            "map_fn": lambda ex: {"anchor": ex["title"], "positive": ex["text"]},
            "cols": Features({"anchor": Value("string"),"positive": Value("string")}),
            "loss": MultipleNegativesRankingLoss
        }
    }

    # Extend with auto-generated StackExchange subsets
    se1 = get_all_data_subset(
        "StackExchange_title_best_answer",
        "flax-sentence-embeddings/stackexchange_title_best_voted_answer_jsonl",
        "title_body", "upvoted_answer", MultipleNegativesRankingLoss
    )
    se2 = get_all_data_subset(
        "StackExchange_titlebody_best_answer",
        "flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl",
        "title_body", "upvoted_answer", MultipleNegativesRankingLoss
    )
    base.update(se1)
    base.update(se2)

    if N_DATA_SRC is not None:
        n_src = min(len(base), N_DATA_SRC)
        base = {k: v for i, (k, v) in enumerate(base.items()) if i < n_src}

    return base


def load_and_cache_datasets(
    configs: dict, 
    CACHE_DIR, 
    NROWS=None, 
    rank=None, 
    streaming: bool = False, 
    val_frac: float = 0.05, 
    test_frac: float = 0.05
) -> dict:
    if rank is not None:
        rank_prefix = f"RANK:{rank};"
    else:
        rank_prefix = ""
    
    out = {}
    n_samples = {}
    for name, cfg in configs.items():
        print(f"{rank_prefix}▶ Processing: {name}")
        cache_path = os.path.join(CACHE_DIR, name) # For non-streaming processed & split dataset

        if not streaming and os.path.isdir(cache_path):
            print(f"{rank_prefix}Loading from disk cache: {cache_path}")
            splits = HFDatasetDict.load_from_disk(cache_path)
            n_samples[name] = sum([_ds.num_rows for s, _ds in splits.items()])
        else:
            load_args = cfg["args"].copy()
            # Initial load
            if streaming:
                print(f"{rank_prefix}Loading (streaming): {name} with args {load_args}")
                raw_ds = load_dataset(**load_args, streaming=streaming, trust_remote_code=True)
                # n_samples[name] = raw_ds.info.splits["train"].num_examples
                
                # If load_dataset returns a dict of streams (e.g. for different configs/splits)
                # The original code concatenates them. For streams, interleave_datasets is an option.
                # Assuming here that each cfg["args"] points to one primary data stream or a dict like {'train': stream}
                if isinstance(raw_ds, dict) or isinstance(raw_ds, HFDatasetDict):
                    n_samples[name] = sum([split_ds.info.splits[split_name].num_examples for split_name, split_ds in raw_ds.items()])
                     # If it's a dict of streams, concat them all
                    raw_ds = next(iter(raw_ds.values())) if isinstance(raw_ds, (dict, HFDatasetDict)) and raw_ds else raw_ds

                if NROWS is not None:
                    raw_ds = raw_ds.take(NROWS)
                    n_samples[name] = NROWS

            else: # Not streaming
                print(f"{rank_prefix}Loading (non-streaming): {name} with args {load_args}")
                # For non-streaming, NROWS can be used to limit the initial load if dataset supports it,
                # or selected afterwards. The original code had a complex NROWS logic here.
                # Simplified: load then select.
                raw_ds_obj = load_dataset(**load_args, trust_remote_code=True)
                if isinstance(raw_ds_obj, HFDatasetDict):
                    raw_ds = concatenate_datasets(list(raw_ds_obj.values()))
                else:
                    raw_ds = raw_ds_obj
                n_samples[name] = raw_ds.num_rows
                if NROWS is not None:
                    n_samples[name] = min(NROWS, len(raw_ds))
                    raw_ds = raw_ds.select(range(n_samples[name]))

            # Map function
            # For IterableDataset, remove_columns in .map() is not directly supported.
            # The map_fn should be structured to return only the desired columns.
            # Also, num_proc is not used in IterableDataset.map().
            print(f"{rank_prefix}Mapping dataset: {name}")
            mapped_ds = raw_ds.map(
                cfg["map_fn"],
                features=cfg["cols"],
                remove_columns=raw_ds.column_names,
                batched=True, # Usually good for map performance
                batch_size=1000 # Adjust as needed
            )

            # Splitting
            if streaming:
                print(f"RANK:{rank}; Columns names after mapping: {mapped_ds.column_names}")
                print("*"*10,mapped_ds )
                # NROWS here is total_samples_if_known for the current stream being processed
                splits = split_iterable_dataset(mapped_ds, val_frac, test_frac, n_samples[name], name=name)
            else: # Not streaming
                # Ensure correct columns are selected if map_fn didn't strictly limit them
                cols_to_select = ["anchor", "positive"]
                if "negative" in mapped_ds.column_names: # Check if 'negative' is present
                    cols_to_select.append("negative")
                mapped_ds = mapped_ds.select_columns(cols_to_select)
                
                splits = original_split_dataset(mapped_ds, val_frac=val_frac, test_frac=test_frac)
                if not os.path.isdir(cache_path): # Save only if not loaded from cache
                    os.makedirs(cache_path, exist_ok=True)
                    print(f"{rank_prefix}Saving processed non-streamed splits to disk: {cache_path}")
                    splits.save_to_disk(cache_path)
        
        out[name] = splits
    return out, n_samples

class MnrLossEvaluator(SentenceEvaluator):
    """
    Evaluates the model based on the MultipleNegativesRankingLoss, calculating loss batch-by-batch.
    Includes EXPLICIT device placement for input tensors as a safeguard.
    """
    def __init__(self, dataloader: DataLoader, name: str = 'mnrl_evaluator', write_csv: bool = False):
        super().__init__()
        if not isinstance(dataloader, DataLoader):
             raise ValueError("dataloader must be a PyTorch DataLoader instance.")
        self.dataloader = dataloader
        self.name = name
        self.primary_metric = f"{name}/avg"
        self.write_csv = write_csv

    def __call__(self,
                 model: SentenceTransformer,
                 output_path: str = None,
                 epoch: int = -1,
                 steps: int = -1) -> float:

        # 1) Setup loss, model & bookkeeping
        loss_fct    = MultipleNegativesRankingLoss(model=model)
        model.eval()
        total_loss  = 0.0
        num_batches = 0

        # 2) Swap in smart‐batching collate if available
        original_collate = self.dataloader.collate_fn
        if hasattr(model, 'smart_batching_collate'):
            self.dataloader.collate_fn = model.smart_batching_collate
        else:
            print(f"Error [{self.name}]: no smart_batching_collate; aborting.")
            return {
                f"{self.name}/avg": float('nan'),
                f"{self.name}/sum": float('nan'),
            }

        # 3) Iterate batches
        for batch_idx, batch in enumerate(self.dataloader):
            # unpack and sanity‐check
            try:
                sentence_features, _ = batch
                if not (isinstance(sentence_features, list) and len(sentence_features) == 2):
                    print(f"Warning [{self.name}]: batch {batch_idx} invalid format; skipping.")
                    continue
            except Exception as e:
                print(f"Warning [{self.name}]: batch {batch_idx} collate error ({e}); skip.")
                continue

            # 4) Move all feature dicts onto model.device
            device = model.device
            sentence_features = [
                { k: tensor.to(device) for k, tensor in feat.items() }
                for feat in sentence_features
            ]

            # 5) Compute loss in one shot (handles encoding + loss)
            with torch.no_grad():
                try:
                    loss = loss_fct(sentence_features, labels=None)
                    total_loss  += loss.item()
                    num_batches += 1
                except Exception as e:
                    print(f"Error [{self.name}]: loss_fct failed on batch {batch_idx}: {e}; skip.")
                    continue

        # 6) Restore original collate
        self.dataloader.collate_fn = original_collate

        # 7) Final stats
        if num_batches == 0:
            print(f"Warning [{self.name}]: no batches processed successfully.")
            return {
                f"{self.name}/avg": float('nan'),
                f"{self.name}/sum": float('nan'),
            }

        average_loss = total_loss / num_batches
        # print(f"[{self.name}] Epoch={epoch} Steps={steps} → avg MNR loss = {average_loss:.4f}")

        # 8) Optional CSV logging
        if output_path and self.write_csv:
            os.makedirs(output_path, exist_ok=True)
            csv_file = os.path.join(output_path, f"{self.name}_results.csv")
            header_needed = not os.path.isfile(csv_file)
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if header_needed:
                    writer.writerow(['epoch', 'steps', 'average_loss'])
                writer.writerow([epoch, steps, average_loss])

        return {
            f"{self.name}/avg": average_loss,
            f"{self.name}/sum": total_loss,
            }
    
def prepare_evaluators(eval_ds: dict, max_per_split: int = 10, BATCH_SIZE: int = 32) -> Optional[SequentialEvaluator]:

    ds_dict_materialized_for_eval = {}

    # Determine if we should take all samples (max_per_split is None or <=0) or a limited number
    take_all_samples = not (max_per_split and max_per_split > 0)

    for k, v_orig in eval_ds.items():
        current_samples_to_take = None
        is_iterable = isinstance(v_orig, IterableDataset)

        if not take_all_samples:
            current_samples_to_take = max_per_split
        
        print(f"Preparing evaluator for '{k}': take_all={take_all_samples}, max_per_split={max_per_split}, current_samples_to_take={current_samples_to_take}, is_iterable={is_iterable}")

        if is_iterable:
            if current_samples_to_take is not None: # Taking a subset
                print(f"Taking {current_samples_to_take} samples from IterableDataset '{k}' for evaluation.")
                samples = list(v_orig.take(current_samples_to_take))
            else: # Taking all
                print(f"Materializing all samples from IterableDataset '{k}' for evaluation. This might be memory intensive or slow.")
                samples = list(v_orig) # Materialize the whole iterable dataset

            if not samples:
                print(f"Warning: IterableDataset '{k}' yielded no samples.")
                # Create an empty dataset with expected schema if possible, or skip
                # For simplicity, we'll skip if no samples, but ideally, schema should be known.
                # ds_dict_materialized_for_eval[k] = HFDataset.from_list([]) # Needs schema
                continue
            
            # Ensure samples are dicts to create Dataset. Assume map_fn produced dicts.
            if isinstance(samples[0], dict):
                # Infer features from the first sample to handle dynamic columns (e.g. +/- "negative")
                ds_dict_materialized_for_eval[k] = HFDataset.from_list(samples)
            else:
                print(f"Warning: Samples from IterableDataset '{k}' are not dicts. Cannot create Dataset for evaluator. Sample type: {type(samples[0])}")
                continue
        
        elif isinstance(v_orig, HFDataset):
            if current_samples_to_take is not None: # Taking a subset
                num_available = len(v_orig)
                actual_to_take = min(current_samples_to_take, num_available)
                print(f"Selecting {actual_to_take} samples from Dataset '{k}' for evaluation.")
                ds_dict_materialized_for_eval[k] = v_orig.select(range(actual_to_take))
            else: # Taking all
                print(f"Using all {len(v_orig)} samples from Dataset '{k}' for evaluation.")
                ds_dict_materialized_for_eval[k] = v_orig
        else:
            print(f"Warning: Unsupported dataset type '{type(v_orig)}' for key '{k}' in prepare_evaluators.")
            continue

    evaluators = []
    all_ir_queries: Dict[str, str] = {}
    all_ir_corpus: Dict[str, str] = {}
    all_ir_rel_docs: Dict[str, set[str]] = {} # Store set of relevant doc IDs for each query ID

    all_triplet_anchors: List[str] = []
    all_triplet_positives: List[str] = []
    all_triplet_negatives: List[str] = []
    
    all_mnrl_samples: List[InputExample] = []

    # --- Process datasets ---
    for ds_name, ds in ds_dict_materialized_for_eval.items():
        try: # Add basic try-except around processing each dataset source
            column_names = ds.column_names
            has_negatives = "negative" in column_names
            has_anchor = "anchor" in column_names
            has_positive = "positive" in column_names

            if not (has_anchor and has_positive):
                print(f"Skipping dataset '{ds_name}': missing 'anchor' or 'positive'.")
                continue

            for i, ex in enumerate(ds):
                anchor = ex["anchor"]
                positive = ex["positive"]

                if not (isinstance(anchor, str) and isinstance(positive, str)):
                     # print(f"Skipping sample {i} in '{ds_name}': anchor or positive is not a string.") # Optional
                     continue

                # --- Data for IR Evaluator ---
                query_key = f"{ds_name}_q{i}"
                corpus_key = f"{ds_name}_c{i}"
                all_ir_queries[query_key] = anchor
                all_ir_corpus[corpus_key] = positive
                # Ensure rel_docs handles multiple relevant docs per query if needed (current setup 1:1)
                if query_key not in all_ir_rel_docs:
                    all_ir_rel_docs[query_key] = set()
                all_ir_rel_docs[query_key].add(corpus_key)


                # --- Data for Triplet Evaluator ---
                if has_negatives:
                    negative = ex.get("negative")
                    if isinstance(negative, str):
                         all_triplet_anchors.append(anchor)
                         all_triplet_positives.append(positive)
                         all_triplet_negatives.append(negative)

                # --- Data for MNRL Evaluator (as InputExample) ---
                all_mnrl_samples.append(InputExample(texts=[anchor, positive]))
        except Exception as e:
            print(f"Error processing dataset source '{ds_name}': {type(e).__name__}: {e}. Skipping this source.")
            continue # Continue to next dataset if one fails


    # --- Create Information Retrieval Evaluator ---
    if all_ir_queries and all_ir_corpus and all_ir_rel_docs:
        try:
            ir_eval = TimedIREvaluator(
                queries=all_ir_queries, corpus=all_ir_corpus, relevant_docs=all_ir_rel_docs,
                name="ir_evaluator", batch_size=BATCH_SIZE,
                mrr_at_k=[1, 5, 10], ndcg_at_k=[1, 5, 10],
                accuracy_at_k=[1, 5, 10], precision_recall_at_k=[1, 5, 10],
                map_at_k=[1, 5, 10],
                show_progress_bar=True, write_csv=True
            )
            evaluators.append(ir_eval)
        except Exception as e:
            print(f"Error creating InformationRetrievalEvaluator: {type(e).__name__}: {e}")

    # --- Create Triplet Evaluator ---
    if all_triplet_anchors:
        try:
            triplet_eval = TripletEvaluator(
                anchors=all_triplet_anchors, positives=all_triplet_positives, negatives=all_triplet_negatives,
                name="triplet_evaluator", batch_size=BATCH_SIZE,
                show_progress_bar=True, write_csv=True
            )
            evaluators.append(triplet_eval)
        except Exception as e:
             print(f"Error creating TripletEvaluator: {type(e).__name__}: {e}")

    # --- Create MNRL Loss Evaluator ---
    if all_mnrl_samples:
        try:
            mnrl_dataloader = DataLoader(
                all_mnrl_samples,
                shuffle=False, # Keep order for evaluation
                batch_size=BATCH_SIZE,
                drop_last=False
            )
            # Instantiate the custom evaluator, passing the dataloader
            mnrl_eval = MnrLossEvaluator(mnrl_dataloader, name="mnr_loss", write_csv=True)
            evaluators.append(mnrl_eval)
        except Exception as e:
            print(f"Error creating MnrLossEvaluator or its DataLoader: {type(e).__name__}: {e}")

    # ... (Rest of prepare_evaluators remains the same) ...
    if not evaluators:
        print("Warning: No evaluators were successfully created.")
        return None

    print(f"Prepared SequentialEvaluator with: {[e.name for e in evaluators]}")
    return SequentialEvaluator(evaluators)