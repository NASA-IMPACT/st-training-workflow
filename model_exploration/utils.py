import argparse
import csv  # For CSV writing
import datetime
import json
import os
import pickle
import random
import sys
import time
from typing import Dict, List, Optional, Union

import distributed
import joblib
import torch
import wandb
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    get_dataset_config_names,
    load_dataset,
    load_from_disk,
)
from distributed import init_ddp, print0
from dotenv import load_dotenv
from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    SentenceTransformerTrainer,
)
from sentence_transformers.data_collator import SentenceTransformerDataCollator
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SentenceEvaluator,
    SequentialEvaluator,
    TripletEvaluator,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import (
    BatchSamplers,
    SentenceTransformerTrainingArguments,
)
from torch import nn
from torch.autograd import Function
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm


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
        "gpu_names": gpu_names,
    }


def split_dataset(
    ds: Union["Dataset", DatasetDict],
    train_split_name: str = "train",
    val_split_name: str = "validation",
    test_split_name: str = "test",
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 42,
) -> DatasetDict:
    """
    Merge all splits, optionally limit rows, then split into train/val/test.
    """
    # Merge splits
    full = (
        concatenate_datasets(list(ds.values())) if isinstance(ds, DatasetDict) else ds
    )
    # Limit rows
    # full = full.select(range(min(NROWS, len(full)))) if NROWS else full

    # Train vs. (Val+Test)
    combined = val_frac + test_frac
    split1 = full.train_test_split(test_size=combined, seed=seed)
    train_ds, eval_ds = split1["train"], split1["test"]

    # Validation vs. Test
    val_rel = val_frac / combined
    split2 = eval_ds.train_test_split(test_size=(1 - val_rel), seed=seed)
    val_ds, test_ds = split2["train"], split2["test"]

    return DatasetDict(
        {
            train_split_name: train_ds,
            val_split_name: val_ds,
            test_split_name: test_ds,
        },
    )


def get_all_data_subset(name: str, path: str, s1: str, s2: str, loss_fn) -> dict:
    """
    Auto-generate configs for all dataset variants under `path`.
    """
    out = {}
    for cfg in get_dataset_config_names(path):
        key = f"{name}_{cfg}"
        out[key] = {
            "args": {"path": path, "name": cfg},
            "map_fn": lambda ex, s1=s1, s2=s2: {"anchor": ex[s1], "positive": ex[s2]},
            "loss": loss_fn,
        }
    return out


def build_dataset_configs_s3(N_DATA_SRC=None) -> dict:
    """
    Define all your dataset mappings and losses for stage 3 data.
    """

    base = {
        "nasa-science-function-code-docstring": {
            "args": {"path": "nasa-impact/nasa-science-function-code-docstring"},
            "map_fn": lambda ex: {
                "anchor": ex["original_docstring"],
                "positive": ex["code"],
            },
            "loss": MultipleNegativesRankingLoss,
            "weight": 1,  # Optional weight for sampling this datadet when using WeightedBatchSampler; if not specified, defaults to 1.0
        },
        "nasa-science-class-code-docstring": {
            "args": {"path": "nasa-impact/nasa-science-class-code-docstring"},
            "map_fn": lambda ex: {
                "anchor": ex["original_docstring"],
                "positive": ex["code"],
            },
            "loss": MultipleNegativesRankingLoss,
            "weight": 1,  # Optional weight for sampling this datadet when using WeightedBatchSampler; if not specified, defaults to 1.0
        },
        "nasa-science-function-code-identifier": {
            "args": {"path": "nasa-impact/nasa-science-function-code-identifier"},
            "map_fn": lambda ex: {"anchor": ex["identifier"], "positive": ex["code"]},
            "loss": MultipleNegativesRankingLoss,
            "weight": 1,  # Optional weight for sampling this datadet when using WeightedBatchSampler; if not specified, defaults to 1.0
        },
        "nasa-science-class-code-identifier": {
            "args": {"path": "nasa-impact/nasa-science-class-code-identifier"},
            "map_fn": lambda ex: {"anchor": ex["identifier"], "positive": ex["code"]},
            "loss": MultipleNegativesRankingLoss,
            "weight": 1,  # Optional weight for sampling this datadet when using WeightedBatchSampler; if not specified, defaults to 1.0
        },
        # "the-vault-function": {
        #     "args": {"path": "Fsoft-AIC/the-vault-function"},
        #     "map_fn": lambda ex: {"anchor": ex["original_docstring"], "positive": ex["code"]},
        #     "loss": MultipleNegativesRankingLoss,
        #     "weight": 1,  # Optional weight for sampling this datadet when using WeightedBatchSampler; if not specified, defaults to 1.0
        # },
        # "the-vault-class": {
        #     "args": {"path": "Fsoft-AIC/the-vault-class"},
        #     "map_fn": lambda ex: {"anchor": ex["original_docstring"], "positive": ex["code"]},
        #     "loss": MultipleNegativesRankingLoss,
        #     "weight": 1,  # Optional weight for sampling this datadet when using WeightedBatchSampler; if not specified, defaults to 1.0
        # },
    }

    if N_DATA_SRC is not None:
        n_src = min(len(base), N_DATA_SRC)
        base = {k: v for i, (k, v) in enumerate(base.items()) if i < n_src}

    return base


def build_dataset_configs_s2(N_DATA_SRC=None) -> dict:
    """
    Define all your dataset mappings and losses for stage 2 data.
    """

    def process_pubmed(example):
        try:
            title = example["MedlineCitation"]["Article"]["ArticleTitle"]
            abstract = example["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
        except (KeyError, TypeError):
            title, abstract = "", ""
        return {
            "anchor": title,
            "positive": abstract,
            # "negative": "",
        }

    base = {
        "nasa-sde-st": {
            "args": {"path": "nasa-impact/nasa-sde-st-corpus"},
            "map_fn": lambda ex: {"anchor": ex["query"], "positive": ex["context"]},
            "loss": MultipleNegativesRankingLoss,
            "weight": 41,  # Optional weight for sampling this datadet when using WeightedBatchSampler; if not specified, defaults to 1.0
        },
        "pubmed_v3": {
            "args": {"path": "../data_prep/raw/pubmed.py", "split": "train"},
            "map_fn": process_pubmed,
            "loss": MultipleNegativesRankingLoss,
            "weight": 41 / 24,
        },
        "arxiv_title_abstract": {
            "args": {
                "path": "json",
                "data_files": "../data_prep/raw/arxiv-metadata-oai-snapshot.json",
            },
            "map_fn": lambda ex: {"anchor": ex["title"], "positive": ex["abstract"]},
            "loss": MultipleNegativesRankingLoss,
            "weight": 41 / 2.7,
        },
        "nasa_ads": {
            "args": {"path": "nasa-impact/nasa_ads_corpus", "data_files": "*.jsonl.gz"},
            "map_fn": lambda ex: {
                "anchor": ex["query"],
                "positive": ex["positives"]["docs"][0],
            },
            "loss": MultipleNegativesRankingLoss,
            "weight": 41 / 2.66,
        },
        "s2orc_title_abstract": {
            "args": {
                "path": "sentence-transformers/s2orc",
                "split": "train",
                "name": "title-abstract-pair",
            },
            "map_fn": lambda ex: {"anchor": ex["title"], "positive": ex["abstract"]},
            "loss": MultipleNegativesRankingLoss,
            "weight": 1,
        },
        # "s2orc_abstract_citation": {
        #     "args": {
        #         "path": "sentence-transformers/s2orc",
        #         "split": "train",
        #         "name": "abstract-citation-pair",
        #     },
        #     "map_fn": lambda ex: {"anchor": ex["abstract"], "positive": ex["citation"]},
        #     "loss": MultipleNegativesRankingLoss,
        # },
        # "s2orc_title_citation": {
        #     "args": {
        #         "path": "sentence-transformers/s2orc",
        #         "split": "train",
        #         "name": "title-citation-pair",
        #     },
        #     "map_fn": lambda ex: {"anchor": ex["title"], "positive": ex["citation"]},
        #     "loss": MultipleNegativesRankingLoss,
        # },
        "specter": {
            "args": {
                "path": "sentence-transformers/specter",
                "split": "train",
                "name": "triplet",
            },
            "map_fn": lambda ex: {
                "anchor": ex["anchor"],
                "positive": ex["positive"],
                "negative": ex["negative"],
            },
            "loss": MultipleNegativesRankingLoss,
            "weight": 41 / 5,
        },
        # "stage1_pairs": {
        #     "args": {"path": "stage1/stage1_pairs"},
        #     "map_fn": lambda ex: {"anchor": ex["query"], "positive": ex["context"]},
        #     "loss": MultipleNegativesRankingLoss,
        #     "weight": 1,  # sample from stage1
        # },
        # "pmc": {
        #     "args": {"path": "../data_prep/raw/pmc_open_access.py", "split": "train"},
        #     "map_fn": lambda ex: {
        #         "anchor": ex["MedlineCitation"]["Article"]["Article Title"],
        #         "positive": ex["MedlineCitation"]["Article"]["Abstract"][
        #             "AbstractText"
        #         ],
        #     },
        #     "loss": MultipleNegativesRankingLoss,
        # },
    }

    if N_DATA_SRC is not None:
        n_src = min(len(base), N_DATA_SRC)
        base = {k: v for i, (k, v) in enumerate(base.items()) if i < n_src}

    return base


def build_dataset_configs_s1(N_DATA_SRC=None) -> dict:
    """
    Define all your dataset mappings and losses.
    """
    base = {
        "squad_v2": {
            "args": {"path": "rajpurkar/squad_v2"},
            "map_fn": lambda ex: {"anchor": ex["question"], "positive": ex["context"]},
            "loss": MultipleNegativesRankingLoss,
        },
        "wikipedia": {
            "args": {"path": "wikimedia/wikipedia", "data_dir": "20231101.en"},
            "map_fn": lambda ex: {"anchor": ex["title"], "positive": ex["text"]},
            "loss": MultipleNegativesRankingLoss,
        },
        "StackExchange_Math_titlebody_answer": {
            "args": {
                "path": "flax-sentence-embeddings/stackexchange_math_jsonl",
                "data_dir": "titlebody_answer",
            },
            "map_fn": lambda ex: {
                "anchor": ex["title"],
                "positive": ex["upvoted_answer"],
            },
            "loss": MultipleNegativesRankingLoss,
        },
        "StackExchange_Math_title_answer": {
            "args": {
                "path": "flax-sentence-embeddings/stackexchange_math_jsonl",
                "data_dir": "title_answer",
            },
            "map_fn": lambda ex: {
                "anchor": ex["title"],
                "positive": ex["upvoted_answer"],
            },
            "loss": MultipleNegativesRankingLoss,
        },
        "StackExchange_title_body": {
            "args": {"path": "flax-sentence-embeddings/stackexchange_title_body_jsonl"},
            "map_fn": lambda ex: {"anchor": ex["texts"][0], "positive": ex["texts"][1]},
            "loss": MultipleNegativesRankingLoss,
        },
        "StackExchange_Duplicates_titlebody_titlebody": {
            "args": {
                "path": "sentence-transformers/stackexchange-duplicates",
                "data_dir": "post-post-pair",
            },
            "map_fn": lambda ex: {"anchor": ex["post1"], "positive": ex["post2"]},
            "loss": MultipleNegativesRankingLoss,
        },
        "StackExchange_Duplicates_body_body": {
            "args": {
                "path": "sentence-transformers/stackexchange-duplicates",
                "data_dir": "body-body-pair",
            },
            "map_fn": lambda ex: {"anchor": ex["body1"], "positive": ex["body2"]},
            "loss": MultipleNegativesRankingLoss,
        },
        "StackExchange_Duplicates_title_title": {
            "args": {
                "path": "sentence-transformers/stackexchange-duplicates",
                "data_dir": "title-title-pair",
            },
            "map_fn": lambda ex: {"anchor": ex["title1"], "positive": ex["title2"]},
            "loss": MultipleNegativesRankingLoss,
        },
        # "WikiAnswer_Pairs": {
        #     "args": {"path": "sentence-transformers/wikianswers-duplicates"},
        #     "map_fn": lambda ex: {"anchor": ex["anchor"], "positive": ex["positive"]},
        #     "loss": MultipleNegativesRankingLoss
        # },
        "Natural_Questions": {
            "args": {"path": "sentence-transformers/natural-questions"},
            "map_fn": lambda ex: {"anchor": ex["query"], "positive": ex["answer"]},
            "loss": MultipleNegativesRankingLoss,
        },
        "PAQ": {
            "args": {"path": "embedding-data/PAQ_pairs"},
            "map_fn": lambda ex: {"anchor": ex["set"][0], "positive": ex["set"][1]},
            "loss": MultipleNegativesRankingLoss,
        },
        "Gooaq": {
            "args": {"path": "sentence-transformers/gooaq"},
            "map_fn": lambda ex: {"anchor": ex["question"], "positive": ex["answer"]},
            "loss": MultipleNegativesRankingLoss,
        },
        "yahoo_title_answers": {
            "args": {
                "path": "sentence-transformers/yahoo-answers",
                "data_dir": "title-answer-pair",
            },
            "map_fn": lambda ex: {"anchor": ex["title"], "positive": ex["answer"]},
            "loss": MultipleNegativesRankingLoss,
        },
        "msmacro_triplet": {
            "args": {
                "path": "sentence-transformers/msmarco-msmarco-MiniLM-L6-v3",
                "data_dir": "triplet-hard",
            },
            "map_fn": lambda ex: {
                "anchor": ex["query"],
                "positive": ex["positive"],
                "negative": ex["negative"],
            },
            "loss": MultipleNegativesRankingLoss,
        },
        "trivia_qa_triplet": {
            "args": {
                "path": "sentence-transformers/trivia-qa-triplet",
                "data_dir": "triplet-all",
            },
            "map_fn": lambda ex: {
                "anchor": ex["anchor"],
                "positive": ex["positive"],
                "negative": ex["negative"],
            },
            "loss": MultipleNegativesRankingLoss,
        },
        "nli_for_simcse_triplet": {
            "args": {
                "path": "sentence-transformers/nli-for-simcse",
                "data_dir": "triplet-all",
            },
            "map_fn": lambda ex: {
                "anchor": ex["anchor"],
                "positive": ex["positive"],
                "negative": ex["negative"],
            },
            "loss": MultipleNegativesRankingLoss,
        },
        "quora_dup_triplet": {
            "args": {
                "path": "sentence-transformers/quora-duplicates",
                "data_dir": "triplet-all",
            },
            "map_fn": lambda ex: {
                "anchor": ex["anchor"],
                "positive": ex["positive"],
                "negative": ex["negative"],
            },
            "loss": MultipleNegativesRankingLoss,
        },
        "WikiAnswers": {
            "args": {"path": "embedding-data/WikiAnswers"},
            "map_fn": lambda ex: dict(
                zip(("anchor", "positive"), random.sample(ex["set"], 2)),
            ),
            "loss": MultipleNegativesRankingLoss,
        },
        "eli5": {
            "args": {"path": "sentence-transformers/eli5"},
            "map_fn": lambda ex: {"anchor": ex["question"], "positive": ex["answer"]},
            "loss": MultipleNegativesRankingLoss,
        },
        "sentence_compression": {
            "args": {"path": "sentence-transformers/sentence-compression"},
            "map_fn": lambda ex: {"anchor": ex["simplified"], "positive": ex["text"]},
            "loss": MultipleNegativesRankingLoss,
        },
        "Flickr30k_Captions": {
            "args": {"path": "sentence-transformers/flickr30k-captions"},
            "map_fn": lambda ex: {"anchor": ex["caption1"], "positive": ex["caption2"]},
            "loss": MultipleNegativesRankingLoss,
        },
        "Coco_Captions": {
            "args": {"path": "sentence-transformers/coco-captions"},
            "map_fn": lambda ex: {"anchor": ex["caption1"], "positive": ex["caption2"]},
            "loss": MultipleNegativesRankingLoss,
        },
        "xsum": {
            "args": {"path": "sentence-transformers/xsum"},
            "map_fn": lambda ex: {"anchor": ex["article"], "positive": ex["summary"]},
            "loss": MultipleNegativesRankingLoss,
        },
        "agnews": {
            "args": {"path": "sentence-transformers/agnews"},
            "map_fn": lambda ex: {"anchor": ex["title"], "positive": ex["description"]},
            "loss": MultipleNegativesRankingLoss,
        },
        "npr": {
            "args": {"path": "sentence-transformers/npr"},
            "map_fn": lambda ex: {"anchor": ex["title"], "positive": ex["body"]},
            "loss": MultipleNegativesRankingLoss,
        },
        "cnn_dailymail": {
            "args": {"path": "abisee/cnn_dailymail", "name": "3.0.0"},
            "map_fn": lambda ex: {
                "anchor": ex["highlights"],
                "positive": ex["article"],
            },
            "loss": MultipleNegativesRankingLoss,
        },
        "cc_news": {
            "args": {"path": "vblagoje/cc_news"},
            "map_fn": lambda ex: {"anchor": ex["title"], "positive": ex["text"]},
            "loss": MultipleNegativesRankingLoss,
        },
    }

    # Extend with auto-generated StackExchange subsets
    se1 = get_all_data_subset(
        "StackExchange_title_best_answer",
        "flax-sentence-embeddings/stackexchange_title_best_voted_answer_jsonl",
        "title_body",
        "upvoted_answer",
        MultipleNegativesRankingLoss,
    )
    se2 = get_all_data_subset(
        "StackExchange_titlebody_best_answer",
        "flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl",
        "title_body",
        "upvoted_answer",
        MultipleNegativesRankingLoss,
    )
    base.update(se1)
    base.update(se2)

    if N_DATA_SRC is not None:
        n_src = min(len(base), N_DATA_SRC)
        base = {k: v for i, (k, v) in enumerate(base.items()) if i < n_src}

    return base


def load_and_cache_datasets(configs: dict, CACHE_DIR, NROWS=None, rank=None) -> dict:
    """
    Load each dataset, map into (anchor, positive[, negative]), split, cache, and return.
    """
    if rank is not None:
        rank = f"RANK:{rank};"
    else:
        rank = ""
    out = {}
    for name, cfg in configs.items():
        print(f"{rank}▶ Processing: {name}")
        cache_path = os.path.join(CACHE_DIR, name)

        if os.path.isdir(cache_path):
            splits = load_from_disk(cache_path, keep_in_memory=True)
        else:
            if NROWS:
                raw = load_dataset(
                    **cfg["args"],
                    split="train[:%d]" % NROWS,
                    trust_remote_code=True,
                    num_proc=max(1, os.cpu_count() // 2),
                )
            else:
                raw = load_dataset(
                    **cfg["args"],
                    trust_remote_code=True,
                    num_proc=max(1, os.cpu_count() // 2),
                )
            raw = (
                concatenate_datasets(list(raw.values()))
                if isinstance(raw, DatasetDict)
                else raw
            )

            mapped = raw.map(
                cfg["map_fn"],
                remove_columns=raw.column_names,
                num_proc=max(1, os.cpu_count() // 2),
            )
            # removed examples with either anchor or positive missing
            mapped = mapped.filter(
                lambda example: example["anchor"] and example["positive"],
            )
            cols = ["anchor", "positive"] + (
                ["negative"] if "negative" in mapped.column_names else []
            )
            mapped = mapped.select_columns(cols)

            splits = split_dataset(mapped)
            os.makedirs(cache_path, exist_ok=True)
            splits.save_to_disk(cache_path)

        # reverseing dataset
        # reversed_indices = range(len(splits["train"]) - 1, -1, -1)
        # splits["train"] = splits["train"].select(reversed_indices)
        splits["train"] = splits[
            "train"
        ].shuffle()  # shuffle the train set so you dont have to skip the already train datapoints

        out[name] = splits
    return out


class MnrLossEvaluator(SentenceEvaluator):
    """
    Evaluates the model based on the MultipleNegativesRankingLoss, calculating loss batch-by-batch.
    Includes EXPLICIT device placement for input tensors as a safeguard.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        name: str = "mnrl_evaluator",
        write_csv: bool = False,
    ):
        super().__init__()
        if not isinstance(dataloader, DataLoader):
            raise ValueError("dataloader must be a PyTorch DataLoader instance.")
        self.dataloader = dataloader
        self.name = name
        self.primary_metric = f"{name}/avg"
        self.write_csv = write_csv

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> float:

        # 1) Setup loss, model & bookkeeping
        loss_fct = MultipleNegativesRankingLoss(model=model)
        model.eval()
        total_loss = 0.0
        num_batches = 0

        # 2) Swap in smart‐batching collate if available
        original_collate = self.dataloader.collate_fn
        if hasattr(model, "smart_batching_collate"):
            self.dataloader.collate_fn = model.smart_batching_collate
        else:
            print(f"Error [{self.name}]: no smart_batching_collate; aborting.")
            return {
                f"{self.name}/avg": float("nan"),
                f"{self.name}/sum": float("nan"),
            }

        # 3) Iterate batches
        for batch_idx, batch in enumerate(self.dataloader):
            # unpack and sanity‐check
            try:
                sentence_features, _ = batch
                if not (
                    isinstance(sentence_features, list) and len(sentence_features) == 2
                ):
                    print(
                        f"Warning [{self.name}]: batch {batch_idx} invalid format; skipping.",
                    )
                    continue
            except Exception as e:
                print(
                    f"Warning [{self.name}]: batch {batch_idx} collate error ({e}); skip.",
                )
                continue

            # 4) Move all feature dicts onto model.device
            device = model.device
            sentence_features = [
                {k: tensor.to(device) for k, tensor in feat.items()}
                for feat in sentence_features
            ]

            # 5) Compute loss in one shot (handles encoding + loss)
            with torch.no_grad():
                try:
                    loss = loss_fct(sentence_features, labels=None)
                    total_loss += loss.item()
                    num_batches += 1
                except Exception as e:
                    print(
                        f"Error [{self.name}]: loss_fct failed on batch {batch_idx}: {e}; skip.",
                    )
                    continue

        # 6) Restore original collate
        self.dataloader.collate_fn = original_collate

        # 7) Final stats
        if num_batches == 0:
            print(f"Warning [{self.name}]: no batches processed successfully.")
            return {
                f"{self.name}/avg": float("nan"),
                f"{self.name}/sum": float("nan"),
            }

        average_loss = total_loss / num_batches
        # print(f"[{self.name}] Epoch={epoch} Steps={steps} → avg MNR loss = {average_loss:.4f}")

        # 8) Optional CSV logging
        if output_path and self.write_csv:
            os.makedirs(output_path, exist_ok=True)
            csv_file = os.path.join(output_path, f"{self.name}_results.csv")
            header_needed = not os.path.isfile(csv_file)
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                if header_needed:
                    writer.writerow(["epoch", "steps", "average_loss"])
                writer.writerow([epoch, steps, average_loss])

        return {
            f"{self.name}/avg": average_loss,
            f"{self.name}/sum": total_loss,
        }


def prepare_evaluators(
    eval_ds: dict,
    max_per_split: int = 10,
    BATCH_SIZE: int = 32,
    cache_dir: str = None,
) -> Optional[SequentialEvaluator]:
    """
    Prepares a SequentialEvaluator by processing evaluation datasets and creating
    various evaluators (Information Retrieval, Triplet, and MNRL Loss evaluators).
    Args:
        eval_ds (dict): A dictionary where keys are dataset names and values are datasets.
                        Each dataset is expected to have columns like 'anchor', 'positive',
                        and optionally 'negative'.
        max_per_split (int, optional): Maximum number of samples to use per dataset split.
                                        If set to a positive value, datasets will be truncated
                                        to this size. Defaults to 10.
        BATCH_SIZE (int, optional): Batch size to use for evaluators. Defaults to 32.
        cache_dir (str, optional): Directory to cache processed datasets and evaluator data.
                                   If None, caching is disabled. Defaults to None.
    Returns:
        Optional[SequentialEvaluator]: A SequentialEvaluator containing the created evaluators.
                                       Returns None if no evaluators were successfully created.
    Raises:
        Exception: Propagates exceptions encountered during dataset processing or evaluator creation.
    Notes:
        - The function processes datasets to prepare data for different types of evaluators:
          Information Retrieval (IR), Triplet, and MNRL Loss evaluators.
        - Processed data is cached to avoid redundant computations.
        - If a dataset lacks required columns ('anchor' and 'positive'), it is skipped.
        - Evaluators are created only if sufficient data is available for their respective types.
        - The function uses multiprocessing for dataset mapping to improve performance.
        - If no evaluators are created, a warning is printed, and the function returns None.
    """

    # --- 1. Truncate datasets if max_per_split is set ---
    if max_per_split and max_per_split > 0:
        # Assume eval_ds.items() and v.select() work, or let errors propagate
        ds_dict = {
            k: v.select(range(min(max_per_split, len(v)))) for k, v in eval_ds.items()
        }
        max_per_split_str = str(max_per_split)
    else:
        max_per_split_str = "None"
        ds_dict = eval_ds

    # --- 2. Set up cache directory only if a path is provided ---
    if cache_dir:
        cache_dir = os.path.join(cache_dir, f"max_per_split_{max_per_split_str}")
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Cache directory is set to: {cache_dir}")
    else:
        print("`cache_dir` is None. Caching is disabled.")

    evaluators = []
    all_ir_queries, all_ir_corpus, all_ir_rel_docs = {}, {}, {}
    all_triplet_anchors, all_triplet_positives, all_triplet_negatives = [], [], []
    all_mnrl_samples = []  # This will hold InputExample objects

    # --- 3. Process datasets ---
    for ds_name, ds in ds_dict.items():
        print("Transforming dataset source for Evaluation:", ds_name)
        processed_data = None
        ds_cache_path = os.path.join(cache_dir, f"{ds_name}.pkl") if cache_dir else None

        # Try to load from cache only if cache_dir and the file exist
        if ds_cache_path and os.path.isfile(ds_cache_path):
            print(f"Loading processed data for '{ds_name}' from cache.")
            processed_data = joblib.load(ds_cache_path)

        else:
            processed_data = {}
            try:  # --- Process the data if not loaded from cache ---
                column_names = ds.column_names
                has_negatives = "negative" in column_names
                has_anchor = "anchor" in column_names
                has_positive = "positive" in column_names

                if not (has_anchor and has_positive):
                    print(
                        f"Skipping dataset '{ds_name}': missing 'anchor' or 'positive'.",
                    )
                    continue

                def add_ids(example, idx):
                    """A function to add qid and cid based on the index."""
                    return {
                        "qid": f"{ds_name}_q_{idx}",
                        "cid": f"{ds_name}_c_{idx}",
                    }

                # --- Data for IR Evaluator ---
                ds = ds.map(
                    add_ids,
                    with_indices=True,
                    num_proc=os.cpu_count() // 2,
                )  # Use multiprocessing for map
                processed_data["all_ir_queries"] = dict(zip(ds["qid"], ds["anchor"]))
                processed_data["all_ir_corpus"] = dict(zip(ds["cid"], ds["positive"]))
                dx = ds.to_pandas()
                positive_to_cids_map = dx.groupby("positive")["cid"].apply(set)
                relevant_sets = dx["positive"].map(positive_to_cids_map)
                processed_data["all_ir_rel_docs"] = dict(zip(dx["qid"], relevant_sets))

                # --- Data for Triplet Evaluator ---
                if has_negatives:
                    # Extend lists with entire columns at once
                    processed_data["all_triplet_anchors"] = ds["anchor"]
                    processed_data["all_triplet_positives"] = ds["positive"]
                    processed_data["all_triplet_negatives"] = ds["negative"]

                # --- Data for MNRL Evaluator (as InputExample) ---
                processed_data["all_mnrl_samples"] = [
                    InputExample(texts=[anchor, positive])
                    for anchor, positive in zip(ds["anchor"], ds["positive"])
                ]

                # Save to cache only if cache_dir is specified
                if ds_cache_path:
                    print(f"Saving processed data for '{ds_name}' to cache.")
                    joblib.dump(processed_data, ds_cache_path)

            except Exception as e:
                print(
                    f"Error processing dataset source '{ds_name}': {type(e).__name__}: {e}. Skipping this source.",
                )
                continue  # Continue to next dataset if one fails

        # --- 4. Aggregate processed data ---
        all_ir_queries.update(processed_data.get("all_ir_queries", {}))
        all_ir_corpus.update(processed_data.get("all_ir_corpus", {}))
        all_ir_rel_docs.update(processed_data.get("all_ir_rel_docs", {}))
        all_triplet_anchors.extend(processed_data.get("all_triplet_anchors", []))
        all_triplet_positives.extend(processed_data.get("all_triplet_positives", []))
        all_triplet_negatives.extend(processed_data.get("all_triplet_negatives", []))
        all_mnrl_samples.extend(processed_data.get("all_mnrl_samples", []))

    # --- Create Information Retrieval Evaluator ---
    if all_ir_queries and all_ir_corpus and all_ir_rel_docs:
        try:
            ir_eval = InformationRetrievalEvaluator(
                queries=all_ir_queries,
                corpus=all_ir_corpus,
                relevant_docs=all_ir_rel_docs,
                name="ir_evaluator",
                batch_size=BATCH_SIZE,
                mrr_at_k=[1, 5, 10],
                ndcg_at_k=[1, 5, 10],
                accuracy_at_k=[1, 5, 10],
                precision_recall_at_k=[1, 5, 10],
                map_at_k=[1, 5, 10],
                corpus_chunk_size=5000,
                show_progress_bar=True,
                write_csv=True,
            )
            evaluators.append(ir_eval)
        except Exception as e:
            print(
                f"Error creating InformationRetrievalEvaluator: {type(e).__name__}: {e}",
            )

    # --- Create Triplet Evaluator ---
    if all_triplet_anchors:
        try:
            triplet_eval = TripletEvaluator(
                anchors=all_triplet_anchors,
                positives=all_triplet_positives,
                negatives=all_triplet_negatives,
                name="triplet_evaluator",
                batch_size=BATCH_SIZE,
                show_progress_bar=True,
                write_csv=True,
            )
            evaluators.append(triplet_eval)
        except Exception as e:
            print(f"Error creating TripletEvaluator: {type(e).__name__}: {e}")

    # --- Create MNRL Loss Evaluator ---
    if all_mnrl_samples:
        try:
            mnrl_dataloader = DataLoader(
                all_mnrl_samples,
                shuffle=False,  # Keep order for evaluation
                batch_size=BATCH_SIZE,
                drop_last=False,
            )
            # Instantiate the custom evaluator, passing the dataloader
            mnrl_eval = MnrLossEvaluator(
                mnrl_dataloader,
                name="mnr_loss",
                write_csv=True,
            )
            evaluators.append(mnrl_eval)
        except Exception as e:
            print(
                f"Error creating MnrLossEvaluator or its DataLoader: {type(e).__name__}: {e}",
            )

    if not evaluators:
        print("Warning: No evaluators were successfully created.")
        return None

    print(f"Prepared SequentialEvaluator with: {[e.name for e in evaluators]}")
    return SequentialEvaluator(evaluators)


class RetrievalTimer:
    def __init__(self, model, corpus, queries, top_k: int = 10, device: str = None):
        """
        model   : a SentenceTransformer or any .encode()-capable model
        corpus  : list of strings (documents)
        queries : list of strings
        top_k   : how many nearest neighbors to retrieve per query
        device  : e.g. 'cpu' or 'cuda'; if None, uses model's default
        """
        self.model = model
        self.corpus = corpus
        self.queries = queries
        self.top_k = top_k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Will be filled in by run()
        self.doc_embeddings = None
        self.encode_times = []
        self.search_times = []
        self.retrieval_times = []  # ← new

    def encode_corpus(self):
        t0 = time.perf_counter()
        self.doc_embeddings = self.model.encode(
            self.corpus,
            convert_to_tensor=True,
            device=self.device,
        )
        self.encode_corpus_time = time.perf_counter() - t0
        return self.encode_corpus_time

    def run(self):
        """Runs timing: corpus encoding + per‑query encode & search."""
        # 1. Encode corpus
        corpus_time = self.encode_corpus()

        # 2. For each query, time encode + search
        for q in self.queries:
            # encode query
            t0 = time.perf_counter()
            q_emb = self.model.encode([q], convert_to_tensor=True, device=self.device)
            t1 = time.perf_counter()

            # search (dot‑product)
            t2 = time.perf_counter()
            scores = torch.matmul(q_emb, self.doc_embeddings.T)[0]
            _ = torch.topk(scores, k=self.top_k)
            t3 = time.perf_counter()

            encode_dt = t1 - t0
            search_dt = t3 - t2
            self.encode_times.append(encode_dt)
            self.search_times.append(search_dt)
            self.retrieval_times.append(
                encode_dt + search_dt,
            )  # ← capture per‑query retrieval

        return {
            "corpus_encode_s": corpus_time,
            "avg_query_encode_s": sum(self.encode_times) / len(self.encode_times),
            "avg_query_search_s": sum(self.search_times) / len(self.search_times),
            "avg_query_retrieval_s": sum(self.retrieval_times)
            / len(self.retrieval_times),  # ← new
            "total_query_time_s": sum(self.retrieval_times),
        }

    def print_report(self, times_dict=None):
        """Nicely print timing summary."""
        td = times_dict or self.run()
        print(f"Corpus encoding time:       {td['corpus_encode_s']:.3f} s")
        print(f"Avg query‑encoding time:    {td['avg_query_encode_s']*1000:.2f} ms")
        print(f"Avg query‑search time:      {td['avg_query_search_s']*1000:.2f} ms")
        print(
            f"Avg query‑retrieval time:   {td['avg_query_retrieval_s']*1000:.2f} ms",
        )  # ← new
        print(
            f"Total per‑query time:       {(td['avg_query_retrieval_s'])*1000:.2f} ms",
        )


class PreTokenizedPyTorchDataset(torch.utils.data.Dataset):
    """
    Wraps a HF Dataset with tokenized columns into a torch Dataset.
    Expects each example to have at least:
      - anchor_input_ids, anchor_attention_mask
      - positive_input_ids, positive_attention_mask
    Optionally:
      - negative_input_ids, negative_attention_mask
    """

    def __init__(self, hf_split):
        self.ds = hf_split

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]

    def __getattr__(self, name):
        # Forward any missing attribute to the HF Dataset
        return getattr(self.ds, name)


class PreTokenizedCollator(SentenceTransformerDataCollator):
    """
    Collate_fn that batches pre‑tokenized anchor/positive pairs,
    and optionally negatives if present.
    Returns:
      - ([anchors, positives], None)        if no negatives
      - ([anchors, positives, negatives], None) if negatives exist
    """

    def __init__(self, tokenize_fn=None, **kwargs):
        # Provide a dummy tokenizer if none is given, as parent expects it.
        _tokenize_fn = tokenize_fn if tokenize_fn is not None else lambda x: x
        super().__init__(tokenize_fn=_tokenize_fn, **kwargs)

    # Trainer inspects this to know “I have no label columns”
    def __call__(self, features):
        column_names = list(features[0].keys())

        # batch: list of examples (each example is a dict of lists of length max_len)
        # stack them into tensors of shape (batch_size, max_len)
        anchor_ids = torch.stack(
            [torch.tensor(ex["anchor_input_ids"], dtype=torch.long) for ex in features],
        )
        anchor_mask = torch.stack(
            [
                torch.tensor(ex["anchor_attention_mask"], dtype=torch.long)
                for ex in features
            ],
        )
        positive_ids = torch.stack(
            [
                torch.tensor(ex["positive_input_ids"], dtype=torch.long)
                for ex in features
            ],
        )
        positive_mask = torch.stack(
            [
                torch.tensor(ex["positive_attention_mask"], dtype=torch.long)
                for ex in features
            ],
        )

        batch = {
            "anchor_input_ids": anchor_ids,
            "anchor_attention_mask": anchor_mask,
            "positive_input_ids": positive_ids,
            "positive_attention_mask": positive_mask,
        }

        if "dataset_name" in column_names:
            column_names.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        if tuple(column_names) not in self._warned_columns:
            self.maybe_warn_about_column_order(column_names)

            # Extract the label column if it exists
        for label_column in self.valid_label_columns:
            if label_column in column_names:
                batch["label"] = torch.tensor([row[label_column] for row in features])
                column_names.remove(label_column)
                break

        # if negatives were provided
        if "negative_input_ids" in features[0]:
            neg_ids = torch.stack(
                [
                    torch.tensor(ex["negative_input_ids"], dtype=torch.long)
                    for ex in features
                ],
            )
            neg_mask = torch.stack(
                [
                    torch.tensor(ex["negative_attention_mask"], dtype=torch.long)
                    for ex in features
                ],
            )
            batch["negative_input_ids"] = neg_ids
            batch["negative_attention_mask"] = neg_mask

        return batch


# --- Define the Straight-Through Estimator (STE) for Binarization ---
# We create a custom autograd function to handle the non-differentiable binarization step.


class BinarizeSTE(Function):
    """
    Implements the Straight-Through Estimator for binarization.
    Forward pass: applies torch.sign() to binarize the input tensor to -1 or 1.
    Backward pass: pretends the function was the identity, passing gradients straight through.
    """

    @staticmethod
    def forward(ctx, input):
        # In the forward pass, we apply the sign function to get {-1, 1}.
        # This is the actual binarization.
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        # In the backward pass, we pass the gradient directly through.
        # This is the "straight-through" part. We treat the function as if it were y=x.
        return grad_output


# --- Create a PyTorch Module for our Binarization Layer ---
# This module will apply our custom STE function. We can easily add this to a model.


class BinarizationLayer(nn.Module):
    def __init__(self):
        super(BinarizationLayer, self).__init__()
        self.binarize_fn = BinarizeSTE.apply

    def forward(self, features):
        features["sentence_embedding"] = self.binarize_fn(
            features["sentence_embedding"],
        )
        return features

    def save(self, output_path):
        """Save the binarization layer configuration"""
        os.makedirs(output_path, exist_ok=True)

        # Save a simple config file
        config = {
            "type": "BinarizationLayer",
            "version": "1.0",
        }

        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config, f)

    @staticmethod
    def load(input_path):
        """Load the binarization layer"""
        return BinarizationLayer()

    def get_config_dict(self):
        """Return configuration dictionary"""
        return {
            "type": "BinarizationLayer",
            "version": "1.0",
        }
