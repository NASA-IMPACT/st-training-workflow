import csv  # For CSV writing
import os
import random
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from itertools import chain
from typing import Dict, List, Optional, Union

import distributed
import torch
import wandb
from datasets import Dataset
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from datasets import DatasetDict as HFDatasetDict
from datasets import (
    Features,
    IterableDataset,
    Sequence,
    Value,
    concatenate_datasets,
    get_dataset_config_names,
    interleave_datasets,
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
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SentenceEvaluator,
    SequentialEvaluator,
    TripletEvaluator,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss as MNRL
from sentence_transformers.training_args import (
    BatchSamplers,
    SentenceTransformerTrainingArguments,
)
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader


class MultipleNegativesRankingLoss(MNRL):
    # overriding the forward function to handle negatives that are empty string <"">
    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: Tensor,
    ) -> Tensor:
        # Compute the embeddings and distribute them to anchor and candidates (positive and optionally negatives)

        embeddings = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]

        anchors = embeddings[0]  # (batch_size, embedding_dim)

        # check for empty negatives
        for i, emb in enumerate(embeddings[2:]):
            all_indices = torch.arange(emb.size(0), device=emb.device)
            attention_mask_sum = sentence_features[i + 2]["attention_mask"].sum(axis=1)
            empty_str_indices = (attention_mask_sum == 2).nonzero(as_tuple=True)[0]
            keep_mask = ~torch.isin(all_indices, empty_str_indices)
            embeddings[2 + i] = emb[keep_mask]

        candidates = torch.cat(
            embeddings[1:],
        )  # (batch_size * (1 + num_negatives), embedding_dim)

        # For every anchor, we compute the similarity to all other candidates (positives and negatives),
        # also from other anchors. This gives us a lot of in-batch negatives.
        scores = self.similarity_fct(anchors, candidates) * self.scale
        # (batch_size, batch_size * (1 + num_negatives))

        # anchor[i] should be most similar to candidates[i], as that is the paired positive,
        # so the label for anchor[i] is i
        range_labels = torch.arange(0, scores.size(0), device=scores.device)

        return self.cross_entropy_loss(scores, range_labels)


class TimedIREvaluator(InformationRetrievalEvaluator):
    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        start = time.perf_counter()
        results = super().__call__(model, output_path, epoch, steps)
        elapsed = time.perf_counter() - start

        n_q = len(self.queries)
        avg = elapsed / n_q
        print(
            f"[TimedIREvaluator] total {elapsed:.2f}s over {n_q} queries → avg {avg*1000:.1f} ms/query",
        )

        # optionally add to the results dict for logging to CSV
        results["retrieval_time_total_s"] = elapsed
        results["retrieval_time_avg_s"] = avg
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
        "gpu_names": gpu_names,
    }


# Keep your original split_dataset for non-streaming cases
def original_split_dataset(
    ds: HFDataset,  # Expects a regular Hugging Face Dataset
    train_split_name: str = "train",
    val_split_name: str = "validation",
    test_split_name: str = "test",
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 42,
) -> HFDatasetDict:
    full = ds
    combined = val_frac + test_frac

    if combined == 0:  # No validation or test set
        return HFDatasetDict(
            {
                train_split_name: full,
                val_split_name: HFDataset.from_dict(
                    {col: [] for col in full.features.keys()},
                    features=full.features,
                ),
                test_split_name: HFDataset.from_dict(
                    {col: [] for col in full.features.keys()},
                    features=full.features,
                ),
            },
        )
    if combined >= 1.0:
        raise ValueError(
            "val_frac + test_frac must be less than 1.0 if both are non-zero",
        )

    split1 = full.train_test_split(test_size=combined, seed=seed, shuffle=True)
    train_ds, eval_ds = split1["train"], split1["test"]

    if val_frac == 0:  # Only test split from eval_ds
        val_ds = HFDataset.from_dict(
            {col: [] for col in eval_ds.features.keys()},
            features=eval_ds.features,
        )
        test_ds = eval_ds
    elif test_frac == 0:  # Only val split from eval_ds
        val_ds = eval_ds
        test_ds = HFDataset.from_dict(
            {col: [] for col in eval_ds.features.keys()},
            features=eval_ds.features,
        )
    else:  # Both val and test
        val_rel = val_frac / combined
        split2 = eval_ds.train_test_split(
            test_size=(1.0 - val_rel),
            seed=seed,
            shuffle=True,
        )  # test_size is for the second part (test)
        val_ds, test_ds = split2["train"], split2["test"]

    split_sizes = {
        "train": len(train_ds),
        "validation": len(val_ds),
        "test": len(test_ds),
    }

    return (
        HFDatasetDict(
            {
                train_split_name: train_ds,
                val_split_name: val_ds,
                test_split_name: test_ds,
            },
        ),
        split_sizes,
    )


def split_iterable_dataset(
    iterable_ds: IterableDataset,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    total_samples_if_known: Optional[int] = None,
    seed: int = 42,
    shuffle_buffer_size: int = 10000,
    name: str = "streamed_dataset",  # For logging
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

    if total_samples_if_known and total_samples_if_known > 0:
        if val_frac + test_frac >= 1.0:
            raise ValueError(
                f"For {name}, val_frac ({val_frac}) + test_frac ({test_frac}) must be < 1.0",
            )
        num_val_samples = int(total_samples_if_known * val_frac)
        num_test_samples = int(total_samples_if_known * test_frac)
        # Train gets the rest
    else:
        if total_samples_if_known == 0:  # Handle case where NROWS might be 0
            num_val_samples = 0
            num_test_samples = 0
        else:
            print(
                f"Warning: total_samples_if_known is None or 0 for streaming split of {name}. Using fixed N (max 1000) for val/test if fractions > 0.",
            )
            num_val_samples = (
                min(1000, int(0.05 * 20000)) if val_frac > 0 else 0
            )  # Default to 5% of 20k or 1k
            num_test_samples = (
                min(1000, int(0.05 * 20000)) if test_frac > 0 else 0
            )  # Default to 5% of 20k or 1k

    print(
        f"IterableDataset split for {name}: val_samples={num_val_samples}, test_samples={num_test_samples}",
    )

    val_ds = shuffled_ds.take(num_val_samples)
    test_ds = shuffled_ds.skip(num_val_samples).take(num_test_samples)
    train_ds = shuffled_ds.skip(num_val_samples + num_test_samples)  # Takes the rest

    split_sizes = {
        "train": total_samples_if_known - num_val_samples - num_test_samples
        if total_samples_if_known
        else None,
        "validation": num_val_samples,
        "test": num_test_samples,
    }

    return {
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    }, split_sizes


def get_all_data_subset(
    name: str,
    path: str,
    s1: str,
    s2: str,
    loss_fn,
    col_union,
    n_rows: dict,
) -> dict:
    """
    Auto-generate configs for all dataset variants under `path`.
    """
    out = {}
    for cfg in get_dataset_config_names(path):
        key = f"{name}_{cfg}"
        out[key] = {
            "args": {"path": path, "name": cfg},
            "map_fn": lambda ex, s1=s1, s2=s2: {
                "anchor": ex[s1],
                "positive": ex[s2],
                "negative": [""] * len(ex[s1]),
            },
            "cols": col_union,
            "loss": loss_fn,
            "total_nrows": n_rows.get(key, None),  # Use provided n_rows or None
        }
    return out


def build_dataset_configs_s2(N_DATA_SRC=None) -> dict:
    """
    Define all your dataset mappings and losses.
    """
    col_union = Features(
        {
            "anchor": Value("string"),
            "positive": Value("string"),
            "negative": Value("string"),
        },
    )

    def process_pubmed_batch(batch):
        """
        Safely processes a batch of PubMed data, handling inconsistent structures
        and filtering out incomplete data points.
        """
        # ====================================================================
        # 1. EXTRACTION - Same as before
        # First, extract all potential data points from the raw batch.
        # ====================================================================
        initial_anchors = []
        initial_positives = []

        for item in batch["MedlineCitation"]:
            citation_dict = None

            # Universal handler for list or dict inconsistency
            if isinstance(item, list):
                if item:
                    citation_dict = item[0]
            elif isinstance(item, dict):
                citation_dict = item

            if not citation_dict:
                initial_anchors.append("")
                initial_positives.append("")
                continue

            # Safely extract Title and Abstract
            title = citation_dict.get("Article", {}).get("ArticleTitle", "")
            initial_anchors.append(title or "")

            abstract_data = (
                citation_dict.get("Article", {}).get("Abstract", {}).get("AbstractText")
            )
            if isinstance(abstract_data, list):
                initial_positives.append(" ".join(abstract_data))
            else:
                initial_positives.append(abstract_data or "")

        # ====================================================================
        # 2. FILTERING - The new logic
        # Now, create the final lists, keeping only pairs where BOTH
        # anchor and positive have content.
        # ====================================================================
        final_anchors = []
        final_positives = []

        for anchor, positive in zip(initial_anchors, initial_positives):
            # The condition: if anchor is not empty AND positive is not empty
            if anchor and positive:
                final_anchors.append(anchor)
                final_positives.append(positive)

        # The 'negative' list should correspond to the final, filtered data.
        final_negatives = [""] * len(final_anchors)

        # ====================================================================
        # 3. RETURN - Return the clean, filtered batch
        # ====================================================================
        return {
            "anchor": final_anchors,
            "positive": final_positives,
            "negative": final_negatives,
        }

    base = {
        # "specter": {
        #     "args": {
        #         "path": "sentence-transformers/specter",
        #         "split": "train",
        #         "name": "triplet",
        #     },
        #     "map_fn": lambda ex: {
        #         "anchor": ex["anchor"],
        #         "positive": ex["positive"],
        #         "negative": ex["negative"],
        #     },
        #     "cols": col_union,
        #     "loss": MultipleNegativesRankingLoss,
        #     "total_nrows": 684_000,  # Approximate number of triplets
        # },
        "pubmed": {
            "args": {"path": "../../data_prep/raw/pubmed.py", "split": "train"},
            "map_fn": process_pubmed_batch,
            "cols": col_union,
            "loss": MultipleNegativesRankingLoss,
            "total_nrows": 24_000_000,  # Approximate number of triplets
        },
        # "arxiv_title_abstract": {
        #     "args": {
        #         "path": "json",
        #         "data_files": "../../data_prep/raw/arxiv-metadata-oai-snapshot.json",
        #     },
        #     "map_fn": lambda ex: {
        #         "anchor": ex["title"],
        #         "positive": ex["abstract"],
        #         "negative": [""] * len(ex["title"]),
        #     },
        #     "cols": col_union,
        #     "loss": MultipleNegativesRankingLoss,
        #     "total_nrows": 2_700_000,
        # },
        # "nasa_ads": {
        #     "args": {"path": "nasa-impact/nasa_ads_corpus", "data_files": "*.jsonl.gz"},
        #     "map_fn": lambda ex: {
        #         "anchor": ex["query"],
        #         "positive": ex["positives"]["docs"][0],
        #         "negative": [""] * len(ex["query"]),
        #     },
        #     "cols": col_union,
        #     "loss": MultipleNegativesRankingLoss,
        #     "total_nrows": 2_660_000,  # Approximate number of triplets
        # },
        # "s2orc_title_abstract": {
        #     "args": {
        #         "path": "sentence-transformers/s2orc",
        #         "split": "train",
        #         "name": "title-abstract-pair",
        #     },
        #     "map_fn": lambda ex: {
        #         "anchor": ex["title"],
        #         "positive": ex["abstract"],
        #         "negative": [""] * len(ex["title"]),
        #     },
        #     "cols": col_union,
        #     "loss": MultipleNegativesRankingLoss,
        #     "total_nrows": 41_800_000,  # Approximate number of pairs
        # },
        # "s2orc_abstract_citation": {
        #     "args": {
        #         "path": "sentence-transformers/s2orc",
        #         "split": "train",
        #         "name": "abstract-citation-pair",
        #     },
        #     "map_fn": lambda ex: {
        #         "anchor": ex["abstract"],
        #         "positive": ex["citation"],
        #         "negative": [""] * len(ex["abstract"]),
        #     },
        #     "cols": col_union,
        #     "loss": MultipleNegativesRankingLoss,
        #     "total_nrows": 39_600_000,  # Approximate number of pairs
        # },
        # "s2orc_title_citation": {
        #     "args": {
        #         "path": "sentence-transformers/s2orc",
        #         "split": "train",
        #         "name": "title-citation-pair",
        #     },
        #     "map_fn": lambda ex: {
        #         "anchor": ex["title"],
        #         "positive": ex["citation"],
        #         "negative": [""] * len(ex["title"]),
        #     },
        #     "loss": MultipleNegativesRankingLoss,
        #     "cols": col_union,
        #     "total_nrows": 51_000_000,  # Approximate number of pairs
        # },
        # "nasa-sde-st": {
        #     "args": {"path": "nasa-impact/nasa-sde-st-corpus"},
        #     "map_fn": lambda ex: {
        #         "anchor": ex["query"],
        #         "positive": ex["context"],
        #         "negative": [""] * len(ex["query"]),
        #     },
        #     "cols": col_union,
        #     "loss": MultipleNegativesRankingLoss,
        #     "total_nrows": 1_100_000,  # Approximate number of pairs
        # },
        # "pmc": {
        #     "args": {"path": "../data_prep/raw/pmc_open_access.py", "split": "train"},
        #     "map_fn": lambda ex: {"anchor": ex["MedlineCitation"]["Article"]["Article Title"], "positive": ex["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]},
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
    col_union = Features(
        {
            "anchor": Value("string"),
            "positive": Value("string"),
            "negative": Value("string"),
        },
    )
    base = {
        "squad_v2": {
            "args": {"path": "rajpurkar/squad_v2"},
            "map_fn": lambda ex: {
                "anchor": ex["question"],
                "positive": ex["context"],
                "negative": [""] * len(ex["question"]),
            },
            "cols": col_union,
            "loss": MultipleNegativesRankingLoss,
            "total_nrows": 142_000,
        },
        "wikipedia": {
            "args": {"path": "wikimedia/wikipedia", "data_dir": "20231101.en"},
            "map_fn": lambda ex: {
                "anchor": ex["title"],
                "positive": ex["text"],
                "negative": [""] * len(ex["title"]),
            },
            "cols": col_union,
            "loss": MultipleNegativesRankingLoss,
            "total_nrows": 6_400_000,  # optional, approx (floor) samples in the daataset if the iterDataset doen't have metadata
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
            "cols": col_union,
            "loss": MultipleNegativesRankingLoss,
            "total_nrows": 52_900_000,
        },
        "StackExchange_Math_titlebody_answer": {
            "args": {
                "path": "flax-sentence-embeddings/stackexchange_math_jsonl",
                "data_dir": "titlebody_answer",
            },
            "map_fn": lambda ex: {
                "anchor": ex["title"],
                "positive": ex["upvoted_answer"],
                "negative": [""] * len(ex["title"]),
            },
            "cols": col_union,
            "loss": MultipleNegativesRankingLoss,
            "total_nrows": 1_100_000,
        },
        "StackExchange_Math_title_answer": {
            "args": {
                "path": "flax-sentence-embeddings/stackexchange_math_jsonl",
                "data_dir": "title_answer",
            },
            "map_fn": lambda ex: {
                "anchor": ex["title"],
                "positive": ex["upvoted_answer"],
                "negative": [""] * len(ex["title"]),
            },
            "cols": col_union,
            "loss": MultipleNegativesRankingLoss,
            "total_nrows": 1_100_000,
        },
        "StackExchange_title_body": {
            "args": {"path": "flax-sentence-embeddings/stackexchange_title_body_jsonl"},
            "map_fn": lambda batch: {
                "anchor": [
                    texts_pair[0]
                    for texts_pair in batch["texts"]
                    if len(texts_pair) >= 2
                ],  # Added safety check
                "positive": [
                    texts_pair[1]
                    for texts_pair in batch["texts"]
                    if len(texts_pair) >= 2
                ],  # Added safety check
                "negative": [
                    "" for texts_pair in batch["texts"] if len(texts_pair) >= 2
                ],  # Added safety check
            },
            "original_cols": Features(
                {
                    "texts": Sequence(feature=Value("string")),
                    "tags": Sequence(feature=Value("string")),
                },
            ),
            "cols": col_union,
            "loss": MultipleNegativesRankingLoss,
            "total_nrows": 5_740_000,
        },
        "StackExchange_Duplicates_titlebody_titlebody": {
            "args": {
                "path": "sentence-transformers/stackexchange-duplicates",
                "data_dir": "post-post-pair",
            },
            "map_fn": lambda ex: {
                "anchor": ex["post1"],
                "positive": ex["post2"],
                "negative": [""] * len(ex["post1"]),
            },
            "cols": col_union,
            "loss": MultipleNegativesRankingLoss,
            "total_nrows": 250_000,
        },
        "StackExchange_Duplicates_body_body": {
            "args": {
                "path": "sentence-transformers/stackexchange-duplicates",
                "data_dir": "body-body-pair",
            },
            "map_fn": lambda ex: {
                "anchor": ex["body1"],
                "positive": ex["body2"],
                "negative": [""] * len(ex["body1"]),
            },
            "cols": col_union,
            "loss": MultipleNegativesRankingLoss,
            "total_nrows": 250_000,
        },
        "StackExchange_Duplicates_title_title": {
            "args": {
                "path": "sentence-transformers/stackexchange-duplicates",
                "data_dir": "title-title-pair",
            },
            "map_fn": lambda ex: {
                "anchor": ex["title1"],
                "positive": ex["title2"],
                "negative": [""] * len(ex["title1"]),
            },
            "cols": col_union,
            "loss": MultipleNegativesRankingLoss,
            "total_nrows": 305_000,
        },
        "Natural_Questions": {
            "args": {"path": "sentence-transformers/natural-questions"},
            "map_fn": lambda ex: {
                "anchor": ex["query"],
                "positive": ex["answer"],
                "negative": [""] * len(ex["query"]),
            },
            "cols": col_union,
            "loss": MultipleNegativesRankingLoss,
            "total_nrows": 100_000,
        },
        "PAQ": {
            "args": {"path": "embedding-data/PAQ_pairs"},
            "map_fn": lambda batch: {
                "anchor": [
                    text_set[0] for text_set in batch["set"] if len(text_set) >= 2
                ],  # Added safety check
                "positive": [
                    text_set[1] for text_set in batch["set"] if len(text_set) >= 2
                ],  # Added safety check
                "negative": [
                    "" for text_set in batch["set"] if len(text_set) >= 2
                ],  # Added safety check
            },
            "original_cols": Features(
                {
                    "set": Sequence(feature=Value("string")),
                },
            ),
            "cols": col_union,
            "loss": MultipleNegativesRankingLoss,
            "total_nrows": 64_000_000,
        },
        "Gooaq": {
            "args": {"path": "sentence-transformers/gooaq"},
            "map_fn": lambda ex: {
                "anchor": ex["question"],
                "positive": ex["answer"],
                "negative": [""] * len(ex["question"]),
            },
            "cols": col_union,
            "loss": MultipleNegativesRankingLoss,
            "total_nrows": 3_000_000,
        },
        "yahoo_title_answers": {
            "args": {
                "path": "sentence-transformers/yahoo-answers",
                "data_dir": "title-answer-pair",
            },
            "map_fn": lambda ex: {
                "anchor": ex["title"],
                "positive": ex["answer"],
                "negative": [""] * len(ex["title"]),
            },
            "cols": col_union,
            "loss": MultipleNegativesRankingLoss,
            "total_nrows": 1_200_000,
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
            "cols": col_union,
            "loss": MultipleNegativesRankingLoss,
            "total_nrows": 13_600_000,
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
            "cols": col_union,
            "loss": MultipleNegativesRankingLoss,
            "total_nrows": 1_900_000,
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
            "cols": col_union,
            "loss": MultipleNegativesRankingLoss,
            "total_nrows": 2_790_000,
        },
        "WikiAnswers": {
            "args": {"path": "embedding-data/WikiAnswers"},
            "map_fn": lambda batch: {
                # This creates a list of dictionaries, then transposes it
                **{
                    k: [
                        dic[k]
                        for dic in (
                            dict(zip(("anchor", "positive"), random.sample(s, 2)))
                            if len(s) >= 2
                            else {
                                "anchor": s[0] if len(s) == 1 else None,
                                "positive": s[0] if len(s) == 1 else None,
                            }  # Handle sets with < 2 items
                            for s in batch["set"]
                        )
                    ]
                    for k in ("anchor", "positive")
                },
                "negative": [""] * len(batch["set"]),
            },
            "original_cols": Features(
                {
                    "set": Sequence(feature=Value("string")),
                },
            ),
            "cols": col_union,
            "loss": MultipleNegativesRankingLoss,
            "total_nrows": 3_200_000,
        },
        "eli5": {
            "args": {"path": "sentence-transformers/eli5"},
            "map_fn": lambda ex: {
                "anchor": ex["question"],
                "positive": ex["answer"],
                "negative": [""] * len(ex["question"]),
            },
            "cols": col_union,
            "loss": MultipleNegativesRankingLoss,
            "total_nrows": 325_000,
        },
        "sentence_compression": {
            "args": {"path": "sentence-transformers/sentence-compression"},
            "map_fn": lambda ex: {
                "anchor": ex["simplified"],
                "positive": ex["text"],
                "negative": [""] * len(ex["simplified"]),
            },
            "cols": col_union,
            "loss": MultipleNegativesRankingLoss,
            "total_nrows": 180_000,
        },
        "Flickr30k_Captions": {
            "args": {"path": "sentence-transformers/flickr30k-captions"},
            "map_fn": lambda ex: {
                "anchor": ex["caption1"],
                "positive": ex["caption2"],
                "negative": [""] * len(ex["caption1"]),
            },
            "cols": col_union,
            "loss": MultipleNegativesRankingLoss,
            "total_nrows": 159_000,
        },
        "Coco_Captions": {
            "args": {"path": "sentence-transformers/coco-captions"},
            "map_fn": lambda ex: {
                "anchor": ex["caption1"],
                "positive": ex["caption2"],
                "negative": [""] * len(ex["caption1"]),
            },
            "cols": col_union,
            "loss": MultipleNegativesRankingLoss,
            "total_nrows": 141_000,
        },
        "xsum": {
            "args": {"path": "sentence-transformers/xsum"},
            "map_fn": lambda ex: {
                "anchor": ex["article"],
                "positive": ex["summary"],
                "negative": [""] * len(ex["article"]),
            },
            "cols": col_union,
            "loss": MultipleNegativesRankingLoss,
            "total_nrows": 227_000,
        },
        "agnews": {
            "args": {"path": "sentence-transformers/agnews"},
            "map_fn": lambda ex: {
                "anchor": ex["title"],
                "positive": ex["description"],
                "negative": [""] * len(ex["title"]),
            },
            "cols": col_union,
            "loss": MultipleNegativesRankingLoss,
            "total_nrows": 1_160_000,
        },
        "npr": {
            "args": {"path": "sentence-transformers/npr"},
            "map_fn": lambda ex: {
                "anchor": ex["title"],
                "positive": ex["body"],
                "negative": [""] * len(ex["title"]),
            },
            "cols": col_union,
            "loss": MultipleNegativesRankingLoss,
            "total_nrows": 594_000,
        },
        "cnn_dailymail": {
            "args": {"path": "abisee/cnn_dailymail", "name": "3.0.0"},
            "map_fn": lambda ex: {
                "anchor": ex["highlights"],
                "positive": ex["article"],
                "negative": [""] * len(ex["highlights"]),
            },
            "cols": col_union,
            "loss": MultipleNegativesRankingLoss,
            "total_nrows": 294_000,
        },
        "cc_news": {
            "args": {"path": "vblagoje/cc_news"},
            "map_fn": lambda ex: {
                "anchor": ex["title"],
                "positive": ex["text"],
                "negative": [""] * len(ex["title"]),
            },
            "cols": col_union,
            "loss": MultipleNegativesRankingLoss,
            "total_nrows": 708_000,
        },
    }

    n_total_rows_se_title_best_answer = {
        "StackExchange_title_best_answer_3dprinting": 3488,
        "StackExchange_title_best_answer_academia": 32137,
        "StackExchange_title_best_answer_ai": 5763,
        "StackExchange_title_best_answer_android": 38077,
        "StackExchange_title_best_answer_anime": 10131,
        "StackExchange_title_best_answer_apple": 92487,
        "StackExchange_title_best_answer_arduino": 16281,
        "StackExchange_title_best_answer_askubuntu": 267135,
        "StackExchange_title_best_answer_astronomy": 9086,
        "StackExchange_title_best_answer_aviation": 18755,
        "StackExchange_title_best_answer_avp": 6450,
        "StackExchange_title_best_answer_beer": 1012,
        "StackExchange_title_best_answer_bicycles": 15708,
        "StackExchange_title_best_answer_bioinformatics": 3135,
        "StackExchange_title_best_answer_biology": 19277,
        "StackExchange_title_best_answer_bitcoin": 22474,
        "StackExchange_title_best_answer_blender": 54153,
        "StackExchange_title_best_answer_boardgames": 11805,
        "StackExchange_title_best_answer_bricks": 3530,
        "StackExchange_title_best_answer_buddhism": 6787,
        "StackExchange_title_best_answer_cardano": 248,
        "StackExchange_title_best_answer_chemistry": 27061,
        "StackExchange_title_best_answer_chess": 6392,
        "StackExchange_title_best_answer_chinese": 8646,
        "StackExchange_title_best_answer_christianity": 11498,
        "StackExchange_title_best_answer_civicrm": 10648,
        "StackExchange_title_best_answer_codegolf": 8211,
        "StackExchange_title_best_answer_codereview": 41748,
        "StackExchange_title_best_answer_coffee": 1188,
        "StackExchange_title_best_answer_cogsci": 5101,
        "StackExchange_title_best_answer_computergraphics": 2306,
        "StackExchange_title_best_answer_conlang": 334,
        "StackExchange_title_best_answer_cooking": 22641,
        "StackExchange_title_best_answer_craftcms": 11236,
        "StackExchange_title_best_answer_crafts": 1659,
        "StackExchange_title_best_answer_crypto": 19404,
        "StackExchange_title_best_answer_cs": 30010,
        "StackExchange_title_best_answer_cseducators": 902,
        "StackExchange_title_best_answer_cstheory": 7742,
        "StackExchange_title_best_answer_datascience": 20503,
        "StackExchange_title_best_answer_dba": 71449,
        "StackExchange_title_best_answer_devops": 3462,
        "StackExchange_title_best_answer_diy": 52896,
        "StackExchange_title_best_answer_drones": 496,
        "StackExchange_title_best_answer_drupal": 67817,
        "StackExchange_title_best_answer_dsp": 17430,
        "StackExchange_title_best_answer_earthscience": 4396,
        "StackExchange_title_best_answer_ebooks": 1107,
        "StackExchange_title_best_answer_economics": 8844,
        "StackExchange_title_best_answer_electronics": 129494,
        "StackExchange_title_best_answer_elementaryos": 5917,
        "StackExchange_title_best_answer_ell": 77892,
        "StackExchange_title_best_answer_emacs": 16830,
        "StackExchange_title_best_answer_engineering": 8649,
        "StackExchange_title_best_answer_english": 100640,
        "StackExchange_title_best_answer_eosio": 1940,
        "StackExchange_title_best_answer_esperanto": 1466,
        "StackExchange_title_best_answer_ethereum": 26124,
        "StackExchange_title_best_answer_expatriates": 4913,
        "StackExchange_title_best_answer_expressionengine": 10742,
        "StackExchange_title_best_answer_fitness": 8297,
        "StackExchange_title_best_answer_freelancing": 1663,
        "StackExchange_title_best_answer_french": 10578,
        "StackExchange_title_best_answer_gamedev": 40154,
        "StackExchange_title_best_answer_gaming": 82887,
        "StackExchange_title_best_answer_gardening": 13246,
        "StackExchange_title_best_answer_genealogy": 2895,
        "StackExchange_title_best_answer_german": 13733,
        "StackExchange_title_best_answer_gis": 100254,
        "StackExchange_title_best_answer_graphicdesign": 28083,
        "StackExchange_title_best_answer_ham": 3501,
        "StackExchange_title_best_answer_hardwarerecs": 2050,
        "StackExchange_title_best_answer_health": 4494,
        "StackExchange_title_best_answer_hermeneutics": 9516,
        "StackExchange_title_best_answer_hinduism": 8999,
        "StackExchange_title_best_answer_history": 10766,
        "StackExchange_title_best_answer_homebrew": 5608,
        "StackExchange_title_best_answer_hsm": 2517,
        "StackExchange_title_best_answer_interpersonal": 3398,
        "StackExchange_title_best_answer_iot": 1359,
        "StackExchange_title_best_answer_iota": 775,
        "StackExchange_title_best_answer_islam": 10052,
        "StackExchange_title_best_answer_italian": 3101,
        "StackExchange_title_best_answer_ja": 17376,
        "StackExchange_title_best_answer_japanese": 20948,
        "StackExchange_title_best_answer_joomla": 5887,
        "StackExchange_title_best_answer_judaism": 26085,
        "StackExchange_title_best_answer_korean": 1406,
        "StackExchange_title_best_answer_languagelearning": 948,
        "StackExchange_title_best_answer_latin": 3969,
        "StackExchange_title_best_answer_law": 16133,
        "StackExchange_title_best_answer_lifehacks": 2576,
        "StackExchange_title_best_answer_linguistics": 6843,
        "StackExchange_title_best_answer_literature": 3539,
        "StackExchange_title_best_answer_magento": 79241,
        "StackExchange_title_best_answer_martialarts": 1737,
        "StackExchange_title_best_answer_materials": 1101,
        "StackExchange_title_best_answer_matheducators": 2706,
        "StackExchange_title_best_answer_mathematica": 59895,
        "StackExchange_title_best_answer_mathoverflow": 85289,
        "StackExchange_title_best_answer_mechanics": 18613,
        "StackExchange_title_best_answer_meta": 1000,
        "StackExchange_title_best_answer_moderators": 504,
        "StackExchange_title_best_answer_monero": 3508,
        "StackExchange_title_best_answer_money": 29404,
        "StackExchange_title_best_answer_movies": 18243,
        "StackExchange_title_best_answer_music": 19936,
        "StackExchange_title_best_answer_musicfans": 2431,
        "StackExchange_title_best_answer_mythology": 1595,
        "StackExchange_title_best_answer_networkengineering": 12590,
        "StackExchange_title_best_answer_opendata": 3842,
        "StackExchange_title_best_answer_opensource": 3221,
        "StackExchange_title_best_answer_or": 1490,
        "StackExchange_title_best_answer_outdoors": 5278,
        "StackExchange_title_best_answer_parenting": 5998,
        "StackExchange_title_best_answer_patents": 3573,
        "StackExchange_title_best_answer_pets": 6156,
        "StackExchange_title_best_answer_philosophy": 13114,
        "StackExchange_title_best_answer_photo": 23204,
        "StackExchange_title_best_answer_physics": 141230,
        "StackExchange_title_best_answer_pm": 5435,
        "StackExchange_title_best_answer_poker": 1665,
        "StackExchange_title_best_answer_politics": 11047,
        "StackExchange_title_best_answer_portuguese": 1964,
        "StackExchange_title_best_answer_pt": 103277,
        "StackExchange_title_best_answer_puzzling": 17448,
        "StackExchange_title_best_answer_quant": 12933,
        "StackExchange_title_best_answer_quantumcomputing": 4320,
        "StackExchange_title_best_answer_raspberrypi": 24143,
        "StackExchange_title_best_answer_retrocomputing": 3907,
        "StackExchange_title_best_answer_reverseengineering": 5817,
        "StackExchange_title_best_answer_robotics": 4648,
        "StackExchange_title_best_answer_rpg": 40435,
        "StackExchange_title_best_answer_ru": 253289,
        "StackExchange_title_best_answer_rus": 16528,
        "StackExchange_title_best_answer_russian": 3937,
        "StackExchange_title_best_answer_salesforce": 87272,
        "StackExchange_title_best_answer_scicomp": 7036,
        "StackExchange_title_best_answer_scifi": 54805,
        "StackExchange_title_best_answer_security": 51355,
        "StackExchange_title_best_answer_serverfault": 238507,
        "StackExchange_title_best_answer_sharepoint": 80420,
        "StackExchange_title_best_answer_sitecore": 7838,
        "StackExchange_title_best_answer_skeptics": 8145,
        "StackExchange_title_best_answer_softwareengineering": 51326,
        "StackExchange_title_best_answer_softwarerecs": 11761,
        "StackExchange_title_best_answer_sound": 8303,
        "StackExchange_title_best_answer_space": 12893,
        "StackExchange_title_best_answer_spanish": 7675,
        "StackExchange_title_best_answer_sports": 4707,
        "StackExchange_title_best_answer_sqa": 9256,
        "StackExchange_title_best_answer_stackapps": 1518,
        "StackExchange_title_best_answer_stats": 115679,
        "StackExchange_title_best_answer_stellar": 1078,
        "StackExchange_title_best_answer_superuser": 352610,
        "StackExchange_title_best_answer_sustainability": 1674,
        "StackExchange_title_best_answer_tex": 171628,
        "StackExchange_title_best_answer_tezos": 1169,
        "StackExchange_title_best_answer_tor": 4167,
        "StackExchange_title_best_answer_travel": 36533,
        "StackExchange_title_best_answer_tridion": 5907,
        "StackExchange_title_best_answer_ukrainian": 1767,
        "StackExchange_title_best_answer_unix": 155414,
        "StackExchange_title_best_answer_ux": 28901,
        "StackExchange_title_best_answer_vegetarianism": 585,
        "StackExchange_title_best_answer_vi": 9000,
        "StackExchange_title_best_answer_webapps": 24867,
        "StackExchange_title_best_answer_webmasters": 30370,
        "StackExchange_title_best_answer_windowsphone": 2807,
        "StackExchange_title_best_answer_woodworking": 2955,
        "StackExchange_title_best_answer_wordpress": 83621,
        "StackExchange_title_best_answer_workplace": 24012,
        "StackExchange_title_best_answer_worldbuilding": 26210,
        "StackExchange_title_best_answer_writers": 9867,
    }
    n_total_rows_se_titlebody_best_answer = {
        "StackExchange_titlebody_best_answer_3dprinting": 3488,
        "StackExchange_titlebody_best_answer_academia": 32137,
        "StackExchange_titlebody_best_answer_ai": 5763,
        "StackExchange_titlebody_best_answer_android": 38077,
        "StackExchange_titlebody_best_answer_anime": 10131,
        "StackExchange_titlebody_best_answer_apple": 92487,
        "StackExchange_titlebody_best_answer_arduino": 16281,
        "StackExchange_titlebody_best_answer_askubuntu": 267135,
        "StackExchange_titlebody_best_answer_astronomy": 9086,
        "StackExchange_titlebody_best_answer_aviation": 18755,
        "StackExchange_titlebody_best_answer_avp": 6450,
        "StackExchange_titlebody_best_answer_beer": 1012,
        "StackExchange_titlebody_best_answer_bicycles": 15708,
        "StackExchange_titlebody_best_answer_bioinformatics": 3135,
        "StackExchange_titlebody_best_answer_biology": 19277,
        "StackExchange_titlebody_best_answer_bitcoin": 22474,
        "StackExchange_titlebody_best_answer_blender": 54153,
        "StackExchange_titlebody_best_answer_boardgames": 11805,
        "StackExchange_titlebody_best_answer_bricks": 3530,
        "StackExchange_titlebody_best_answer_buddhism": 6787,
        "StackExchange_titlebody_best_answer_cardano": 248,
        "StackExchange_titlebody_best_answer_chemistry": 27061,
        "StackExchange_titlebody_best_answer_chess": 6392,
        "StackExchange_titlebody_best_answer_chinese": 8646,
        "StackExchange_titlebody_best_answer_christianity": 11498,
        "StackExchange_titlebody_best_answer_civicrm": 10648,
        "StackExchange_titlebody_best_answer_codegolf": 8211,
        "StackExchange_titlebody_best_answer_codereview": 41748,
        "StackExchange_titlebody_best_answer_coffee": 1188,
        "StackExchange_titlebody_best_answer_cogsci": 5101,
        "StackExchange_titlebody_best_answer_computergraphics": 2306,
        "StackExchange_titlebody_best_answer_conlang": 334,
        "StackExchange_titlebody_best_answer_cooking": 22641,
        "StackExchange_titlebody_best_answer_craftcms": 11236,
        "StackExchange_titlebody_best_answer_crafts": 1659,
        "StackExchange_titlebody_best_answer_crypto": 19404,
        "StackExchange_titlebody_best_answer_cs": 30010,
        "StackExchange_titlebody_best_answer_cseducators": 902,
        "StackExchange_titlebody_best_answer_cstheory": 7742,
        "StackExchange_titlebody_best_answer_datascience": 20503,
        "StackExchange_titlebody_best_answer_dba": 71449,
        "StackExchange_titlebody_best_answer_devops": 3462,
        "StackExchange_titlebody_best_answer_diy": 52896,
        "StackExchange_titlebody_best_answer_drones": 496,
        "StackExchange_titlebody_best_answer_drupal": 67817,
        "StackExchange_titlebody_best_answer_dsp": 17430,
        "StackExchange_titlebody_best_answer_earthscience": 4396,
        "StackExchange_titlebody_best_answer_ebooks": 1107,
        "StackExchange_titlebody_best_answer_economics": 8844,
        "StackExchange_titlebody_best_answer_electronics": 129494,
        "StackExchange_titlebody_best_answer_elementaryos": 5917,
        "StackExchange_titlebody_best_answer_ell": 77892,
        "StackExchange_titlebody_best_answer_emacs": 16830,
        "StackExchange_titlebody_best_answer_engineering": 8649,
        "StackExchange_titlebody_best_answer_english": 100640,
        "StackExchange_titlebody_best_answer_eosio": 1940,
        "StackExchange_titlebody_best_answer_esperanto": 1466,
        "StackExchange_titlebody_best_answer_ethereum": 26124,
        "StackExchange_titlebody_best_answer_expatriates": 4913,
        "StackExchange_titlebody_best_answer_expressionengine": 10742,
        "StackExchange_titlebody_best_answer_fitness": 8297,
        "StackExchange_titlebody_best_answer_freelancing": 1663,
        "StackExchange_titlebody_best_answer_french": 10578,
        "StackExchange_titlebody_best_answer_gamedev": 40154,
        "StackExchange_titlebody_best_answer_gaming": 82887,
        "StackExchange_titlebody_best_answer_gardening": 13246,
        "StackExchange_titlebody_best_answer_genealogy": 2895,
        "StackExchange_titlebody_best_answer_german": 13733,
        "StackExchange_titlebody_best_answer_gis": 100254,
        "StackExchange_titlebody_best_answer_graphicdesign": 28083,
        "StackExchange_titlebody_best_answer_ham": 3501,
        "StackExchange_titlebody_best_answer_hardwarerecs": 2050,
        "StackExchange_titlebody_best_answer_health": 4494,
        "StackExchange_titlebody_best_answer_hermeneutics": 9516,
        "StackExchange_titlebody_best_answer_hinduism": 8999,
        "StackExchange_titlebody_best_answer_history": 10766,
        "StackExchange_titlebody_best_answer_homebrew": 5608,
        "StackExchange_titlebody_best_answer_hsm": 2517,
        "StackExchange_titlebody_best_answer_interpersonal": 3398,
        "StackExchange_titlebody_best_answer_iot": 1359,
        "StackExchange_titlebody_best_answer_iota": 775,
        "StackExchange_titlebody_best_answer_islam": 10052,
        "StackExchange_titlebody_best_answer_italian": 3101,
        "StackExchange_titlebody_best_answer_ja": 17376,
        "StackExchange_titlebody_best_answer_japanese": 20948,
        "StackExchange_titlebody_best_answer_joomla": 5887,
        "StackExchange_titlebody_best_answer_judaism": 26085,
        "StackExchange_titlebody_best_answer_korean": 1406,
        "StackExchange_titlebody_best_answer_languagelearning": 948,
        "StackExchange_titlebody_best_answer_latin": 3969,
        "StackExchange_titlebody_best_answer_law": 16133,
        "StackExchange_titlebody_best_answer_lifehacks": 2576,
        "StackExchange_titlebody_best_answer_linguistics": 6843,
        "StackExchange_titlebody_best_answer_literature": 3539,
        "StackExchange_titlebody_best_answer_magento": 79241,
        "StackExchange_titlebody_best_answer_martialarts": 1737,
        "StackExchange_titlebody_best_answer_materials": 1101,
        "StackExchange_titlebody_best_answer_matheducators": 2706,
        "StackExchange_titlebody_best_answer_mathematica": 59895,
        "StackExchange_titlebody_best_answer_mathoverflow": 85289,
        "StackExchange_titlebody_best_answer_mechanics": 18613,
        "StackExchange_titlebody_best_answer_meta": 1000,
        "StackExchange_titlebody_best_answer_moderators": 504,
        "StackExchange_titlebody_best_answer_monero": 3508,
        "StackExchange_titlebody_best_answer_money": 29404,
        "StackExchange_titlebody_best_answer_movies": 18243,
        "StackExchange_titlebody_best_answer_music": 19936,
        "StackExchange_titlebody_best_answer_musicfans": 2431,
        "StackExchange_titlebody_best_answer_mythology": 1595,
        "StackExchange_titlebody_best_answer_networkengineering": 12590,
        "StackExchange_titlebody_best_answer_opendata": 3842,
        "StackExchange_titlebody_best_answer_opensource": 3221,
        "StackExchange_titlebody_best_answer_or": 1490,
        "StackExchange_titlebody_best_answer_outdoors": 5278,
        "StackExchange_titlebody_best_answer_parenting": 5998,
        "StackExchange_titlebody_best_answer_patents": 3573,
        "StackExchange_titlebody_best_answer_pets": 6156,
        "StackExchange_titlebody_best_answer_philosophy": 13114,
        "StackExchange_titlebody_best_answer_photo": 23204,
        "StackExchange_titlebody_best_answer_physics": 141230,
        "StackExchange_titlebody_best_answer_pm": 5435,
        "StackExchange_titlebody_best_answer_poker": 1665,
        "StackExchange_titlebody_best_answer_politics": 11047,
        "StackExchange_titlebody_best_answer_portuguese": 1964,
        "StackExchange_titlebody_best_answer_pt": 103277,
        "StackExchange_titlebody_best_answer_puzzling": 17448,
        "StackExchange_titlebody_best_answer_quant": 12933,
        "StackExchange_titlebody_best_answer_quantumcomputing": 4320,
        "StackExchange_titlebody_best_answer_raspberrypi": 24143,
        "StackExchange_titlebody_best_answer_retrocomputing": 3907,
        "StackExchange_titlebody_best_answer_reverseengineering": 5817,
        "StackExchange_titlebody_best_answer_robotics": 4648,
        "StackExchange_titlebody_best_answer_rpg": 40435,
        "StackExchange_titlebody_best_answer_ru": 253289,
        "StackExchange_titlebody_best_answer_rus": 16528,
        "StackExchange_titlebody_best_answer_russian": 3937,
        "StackExchange_titlebody_best_answer_salesforce": 87272,
        "StackExchange_titlebody_best_answer_scicomp": 7036,
        "StackExchange_titlebody_best_answer_scifi": 54805,
        "StackExchange_titlebody_best_answer_security": 51355,
        "StackExchange_titlebody_best_answer_serverfault": 238507,
        "StackExchange_titlebody_best_answer_sharepoint": 80420,
        "StackExchange_titlebody_best_answer_sitecore": 7838,
        "StackExchange_titlebody_best_answer_skeptics": 8145,
        "StackExchange_titlebody_best_answer_softwareengineering": 51326,
        "StackExchange_titlebody_best_answer_softwarerecs": 11761,
        "StackExchange_titlebody_best_answer_sound": 8303,
        "StackExchange_titlebody_best_answer_space": 12893,
        "StackExchange_titlebody_best_answer_spanish": 7675,
        "StackExchange_titlebody_best_answer_sports": 4707,
        "StackExchange_titlebody_best_answer_sqa": 9256,
        "StackExchange_titlebody_best_answer_stackapps": 1518,
        "StackExchange_titlebody_best_answer_stats": 115679,
        "StackExchange_titlebody_best_answer_stellar": 1078,
        "StackExchange_titlebody_best_answer_superuser": 352610,
        "StackExchange_titlebody_best_answer_sustainability": 1674,
        "StackExchange_titlebody_best_answer_tex": 171628,
        "StackExchange_titlebody_best_answer_tezos": 1169,
        "StackExchange_titlebody_best_answer_tor": 4167,
        "StackExchange_titlebody_best_answer_travel": 36533,
        "StackExchange_titlebody_best_answer_tridion": 5907,
        "StackExchange_titlebody_best_answer_ukrainian": 1767,
        "StackExchange_titlebody_best_answer_unix": 155414,
        "StackExchange_titlebody_best_answer_ux": 28901,
        "StackExchange_titlebody_best_answer_vegetarianism": 585,
        "StackExchange_titlebody_best_answer_vi": 9000,
        "StackExchange_titlebody_best_answer_webapps": 24867,
        "StackExchange_titlebody_best_answer_webmasters": 30370,
        "StackExchange_titlebody_best_answer_windowsphone": 2807,
        "StackExchange_titlebody_best_answer_woodworking": 2955,
        "StackExchange_titlebody_best_answer_wordpress": 83621,
        "StackExchange_titlebody_best_answer_workplace": 24012,
        "StackExchange_titlebody_best_answer_worldbuilding": 26210,
        "StackExchange_titlebody_best_answer_writers": 9867,
    }
    # Extend with auto-generated StackExchange subsets
    se1 = get_all_data_subset(
        "StackExchange_title_best_answer",
        "flax-sentence-embeddings/stackexchange_title_best_voted_answer_jsonl",
        "title_body",
        "upvoted_answer",
        MultipleNegativesRankingLoss,
        col_union=col_union,
        n_rows=n_total_rows_se_title_best_answer,
    )
    se2 = get_all_data_subset(
        "StackExchange_titlebody_best_answer",
        "flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl",
        "title_body",
        "upvoted_answer",
        MultipleNegativesRankingLoss,
        col_union=col_union,
        n_rows=n_total_rows_se_titlebody_best_answer,
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
    test_frac: float = 0.05,
) -> dict:
    if rank is not None:
        rank_prefix = f"RANK:{rank};"
    else:
        rank_prefix = ""

    train_ds = []
    other_ds = {}
    n_samples = defaultdict(dict)
    for name, cfg in configs.items():
        # looping through dataset sources
        print(f"{rank_prefix}▶ Processing: {name}")
        cache_path = os.path.join(
            CACHE_DIR,
            name,
        )  # For non-streaming processed & split dataset

        if not streaming and os.path.isdir(cache_path):
            # if data is not streamming and already cached, load from disk
            print(f"{rank_prefix}Loading from disk cache: {cache_path}")
            splits = HFDatasetDict.load_from_disk(cache_path)
            n_samples[name] = sum([_ds.num_rows for s, _ds in splits.items()])
        else:
            load_args = cfg["args"].copy()
            # Initial load
            if streaming:
                print(f"{rank_prefix}Loading (streaming): {name} with args {load_args}")
                raw_ds = load_dataset(
                    **load_args,
                    streaming=streaming,
                    trust_remote_code=True,
                    features=cfg.get("original_cols"),
                )

                # If load_dataset returns a dict of streams (e.g. for different configs/splits)
                # The original code concatenates them. For streams, interleave_datasets is an option.
                # Assuming here that each cfg["args"] points to one primary data stream or a dict like {'train': stream}
                if isinstance(raw_ds, dict) or isinstance(raw_ds, HFDatasetDict):
                    try:
                        n_samples[name]["total"] = sum(
                            [
                                split_ds.info.splits.get(split_name).num_examples
                                for split_name, split_ds in raw_ds.items()
                            ],
                        )
                    except AttributeError:
                        # If the dataset does not have .info.splits, fallback to a different method
                        n_samples[name]["total"] = cfg.get("total_nrows")
                        if n_samples[name]["total"] is None:
                            raise AttributeError(
                                f"Can get the total number of samples from the dataset: {name}.",
                            )
                    # If it's a dict of streams, concat them all
                    raw_ds = (
                        next(iter(raw_ds.values()))
                        if isinstance(raw_ds, (dict, HFDatasetDict)) and raw_ds
                        else raw_ds
                    )
                else:  # if the raw_ds is HFDataset
                    try:
                        n_samples[name]["total"] = raw_ds.info.splits[
                            "train"
                        ].num_examples
                    except:
                        n_samples[name]["total"] = cfg.get("total_nrows")

                if NROWS is not None:
                    raw_ds = raw_ds.take(NROWS)
                    n_samples[name]["total"] = NROWS

            else:  # Not streaming
                print(
                    f"{rank_prefix}Loading (non-streaming): {name} with args {load_args}",
                )
                # For non-streaming, NROWS can be used to limit the initial load if dataset supports it,
                # or selected afterwards. The original code had a complex NROWS logic here.
                # Simplified: load then select.
                raw_ds_obj = load_dataset(**load_args, trust_remote_code=True)
                if isinstance(raw_ds_obj, HFDatasetDict):
                    raw_ds = concatenate_datasets(list(raw_ds_obj.values()))
                else:
                    raw_ds = raw_ds_obj
                n_samples[name]["total"] = raw_ds.num_rows
                if NROWS is not None:
                    n_samples[name]["total"] = min(NROWS, len(raw_ds))
                    raw_ds = raw_ds.select(range(n_samples[name]["total"]))

            # Map function
            # For IterableDataset, remove_columns in .map() is not directly supported.
            # The map_fn should be structured to return only the desired columns.
            # Also, num_proc is not used in IterableDataset.map().
            print(f"{rank_prefix}Mapping dataset: {name}")
            mapped_ds = raw_ds.map(
                cfg["map_fn"],
                features=cfg["cols"],
                remove_columns=raw_ds.column_names,
                batched=True,  # Usually good for map performance
                batch_size=1000,  # Adjust as needed
            )

            # Splitting
            if streaming:
                print(
                    f"RANK:{rank}; Columns names after mapping: {mapped_ds.column_names}",
                )
                # NROWS here is total_samples_if_known for the current stream being processed
                splits, split_sizes = split_iterable_dataset(
                    mapped_ds,
                    val_frac,
                    test_frac,
                    n_samples[name]["total"],
                    name=name,
                )
            else:  # Not streaming
                # Ensure correct columns are selected if map_fn didn't strictly limit them
                cols_to_select = ["anchor", "positive"]
                if (
                    "negative" in mapped_ds.column_names
                ):  # Check if 'negative' is present
                    cols_to_select.append("negative")
                mapped_ds = mapped_ds.select_columns(cols_to_select)

                splits, split_sizes = original_split_dataset(
                    mapped_ds,
                    val_frac=val_frac,
                    test_frac=test_frac,
                )
                if not os.path.isdir(cache_path):  # Save only if not loaded from cache
                    os.makedirs(cache_path, exist_ok=True)
                    print(
                        f"{rank_prefix}Saving processed non-streamed splits to disk: {cache_path}",
                    )
                    splits.save_to_disk(cache_path)

            n_samples[name] = {**n_samples[name], **split_sizes}

        for split_name, split_ds in splits.items():
            if split_name == "train":
                train_ds.append(split_ds)
            else:
                # for other data split types, use different data structure
                # name is src name here
                other_ds[name] = other_ds.get(name, {})
                other_ds[name][split_name] = split_ds

    # Concatenate data from different sources for trainds
    if len(train_ds) > 1:
        train_ds = interleave_datasets(train_ds, stopping_strategy="all_exhausted")
        # train_ds = IterableDataset.from_generator(lambda: chain.from_iterable(ds for ds in train_ds))
    else:
        train_ds = train_ds[0] if train_ds else None

    print(train_ds)

    return train_ds, other_ds, n_samples


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
) -> Optional[SequentialEvaluator]:
    # eval_ds is a dict of datasets
    # eval_ds = {
    #     "data source 1": IterDataset or Dataset, ..
    # }

    ds_dict_materialized_for_eval = {}

    # Determine if we should take all samples (max_per_split is None or <=0) or a limited number
    take_all_samples = not (max_per_split and max_per_split > 0)

    for k, v_orig in eval_ds.items():
        current_samples_to_take = None
        is_iterable = isinstance(v_orig, IterableDataset)

        if not take_all_samples:
            current_samples_to_take = max_per_split

        print(
            f"Preparing evaluator for '{k}': take_all={take_all_samples}, max_per_split={max_per_split}, current_samples_to_take={current_samples_to_take}, is_iterable={is_iterable}",
        )

        if is_iterable:
            if current_samples_to_take is not None:  # Taking a subset
                print(
                    f"Taking {current_samples_to_take} samples from IterableDataset '{k}' for evaluation.",
                )
                samples = list(v_orig.take(current_samples_to_take))
            else:  # Taking all
                print(
                    f"Materializing all samples from IterableDataset '{k}' for evaluation. This might be memory intensive or slow.",
                )
                samples = list(v_orig)  # Materialize the whole iterable dataset

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
                print(
                    f"Warning: Samples from IterableDataset '{k}' are not dicts. Cannot create Dataset for evaluator. Sample type: {type(samples[0])}",
                )
                continue

        elif isinstance(v_orig, HFDataset):
            if current_samples_to_take is not None:  # Taking a subset
                num_available = len(v_orig)
                actual_to_take = min(current_samples_to_take, num_available)
                print(
                    f"Selecting {actual_to_take} samples from Dataset '{k}' for evaluation.",
                )
                ds_dict_materialized_for_eval[k] = v_orig.select(range(actual_to_take))
            else:  # Taking all
                print(
                    f"Using all {len(v_orig)} samples from Dataset '{k}' for evaluation.",
                )
                ds_dict_materialized_for_eval[k] = v_orig
        else:
            print(
                f"Warning: Unsupported dataset type '{type(v_orig)}' for key '{k}' in prepare_evaluators.",
            )
            continue

    evaluators = []
    all_ir_queries: Dict[str, str] = {}
    all_ir_corpus: Dict[str, str] = {}
    all_ir_rel_docs: Dict[
        str,
        set[str],
    ] = {}  # Store set of relevant doc IDs for each query ID

    all_triplet_anchors: List[str] = []
    all_triplet_positives: List[str] = []
    all_triplet_negatives: List[str] = []

    all_mnrl_samples: List[InputExample] = []

    # --- Process datasets ---
    for ds_name, ds in ds_dict_materialized_for_eval.items():
        try:  # Add basic try-except around processing each dataset source
            column_names = ds.column_names
            has_negatives = "negative" in column_names and ds["negative"][0] != ""
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
            print(
                f"Error processing dataset source '{ds_name}': {type(e).__name__}: {e}. Skipping this source.",
            )
            continue  # Continue to next dataset if one fails

    # --- Create Information Retrieval Evaluator ---
    if all_ir_queries and all_ir_corpus and all_ir_rel_docs:
        try:
            ir_eval = TimedIREvaluator(
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

    # ... (Rest of prepare_evaluators remains the same) ...
    if not evaluators:
        print("Warning: No evaluators were successfully created.")
        return None

    print(f"Prepared SequentialEvaluator with: {[e.name for e in evaluators]}")
    return SequentialEvaluator(evaluators)
