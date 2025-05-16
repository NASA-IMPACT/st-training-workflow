import os
from datasets import load_dataset, DatasetDict, concatenate_datasets, get_dataset_config_names, load_from_disk
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from typing import Optional, Union
from sentence_transformers.training_args import SentenceTransformerTrainingArguments, BatchSamplers
from sentence_transformers.evaluation import InformationRetrievalEvaluator, TripletEvaluator, SequentialEvaluator


NROWS = None
val_frac = 0.05
test_frac = 0.05

CACHE_DIR = f"../data/stage1_cache/NROWS_{NROWS}"
os.makedirs(CACHE_DIR, exist_ok=True)

def split_dataset(
    ds,
    train_split_name: str = "train",
    val_split_name: str = "validation",
    test_split_name: str = "test",
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 42
) -> DatasetDict:
    """
    Always merge all existing splits (if any) into one dataset,
    then carve out a new train/validation split.
    """
    # 1. Merge everything into a single Dataset
    if isinstance(ds, DatasetDict):
        # concatenate all splits (train, validation, test, etc.)
        full_ds = concatenate_datasets(list(ds.values()))
    else:
        # already a single Dataset
        full_ds = ds
        
    # 1.1 Optionally limit the number of rows
    if NROWS:
        # only load a small subset of the dataset
        full_ds = full_ds.select(range(min(NROWS, len(full_ds))))

    # 2) train vs. combined eval
    combined_frac = val_frac + test_frac
    split1 = full_ds.train_test_split(test_size=combined_frac, seed=seed)
    train_ds = split1["train"]
    eval_ds  = split1["test"]

    # 3) validation vs. test
    #    compute relative fraction for validation within eval_ds
    val_relative = val_frac / combined_frac
    split2 = eval_ds.train_test_split(test_size=(1.0 - val_relative), seed=seed)
    val_ds  = split2["train"]
    test_ds = split2["test"]

    return DatasetDict({
        train_split_name: train_ds,
        val_split_name:   val_ds,
        test_split_name:  test_ds,
    })


def get_all_data_subset(name, path, s1, s2, loss_fn):
    configs = get_dataset_config_names(path)
    respo = []

    for cfg in configs:
        _t = {
                "args": {"path": path, "name": cfg},
                "map_fn": lambda ex, s1=s1, s2=s2: {"anchor": ex[s1], "positive": ex[s2]},
                "loss": loss_fn
            }
        respo.append(_t)

    return {f"{name}_{c}": r for r, c in zip(respo, configs)}


dataset_names = {
    "squad_v2": {
        "args": {"path": "rajpurkar/squad_v2"},
        "map_fn": lambda ex: {"anchor": ex["question"], "positive": ex["context"]},
        "loss": MultipleNegativesRankingLoss
    },
    "wikipedia": {
        "args": {"path": "wikimedia/wikipedia", "data_dir": "20231101.en"},
        "map_fn": lambda ex: {"anchor": ex["title"], "positive": ex["text"]},
        "loss": MultipleNegativesRankingLoss
    },
    "StackExchange_Math_titlebody_answer": {
        "args": {"path": "flax-sentence-embeddings/stackexchange_math_jsonl", "data_dir": "titlebody_answer"},
        "map_fn": lambda ex: {"anchor": ex["title"], "positive": ex["upvoted_answer"]},
        "loss": MultipleNegativesRankingLoss
    },
    "StackExchange_Math_title_answer": {
        "args": {"path": "flax-sentence-embeddings/stackexchange_math_jsonl", "data_dir": "title_answer"},
        "map_fn": lambda ex: {"anchor": ex["title"], "positive": ex["upvoted_answer"]},
        "loss": MultipleNegativesRankingLoss
    },
    "StackExchange_title_body": {
        "args": {"path": "flax-sentence-embeddings/stackexchange_title_body_jsonl"},
        "map_fn": lambda ex: {"anchor": ex["texts"][0], "positive": ex["texts"][1]},
        "loss": MultipleNegativesRankingLoss
    },
    "StackExchange_Duplicates_titlebody_titlebody": {
        "args": {"path": "sentence-transformers/stackexchange-duplicates", "data_dir": "post-post-pair"},
        "map_fn": lambda ex: {"anchor": ex["post1"], "positive": ex["post2"]},
        "loss": MultipleNegativesRankingLoss
    },
    "StackExchange_Duplicates_body_body": {
        "args": {"path": "sentence-transformers/stackexchange-duplicates", "data_dir": "body-body-pair"},
        "map_fn": lambda ex: {"anchor": ex["body1"], "positive": ex["body2"]},
        "loss": MultipleNegativesRankingLoss
    },
    "StackExchange_Duplicates_title_title": {
        "args": {"path": "sentence-transformers/stackexchange-duplicates", "data_dir": "title-title-pair"},
        "map_fn": lambda ex: {"anchor": ex["title1"], "positive": ex["title2"]},
        "loss": MultipleNegativesRankingLoss
    },
    "WikiAnswer_Pairs": {
        "args": {"path": "sentence-transformers/wikianswers-duplicates"},
        "map_fn": lambda ex: {"anchor": ex["anchor"], "positive": ex["positive"]},
        "loss": MultipleNegativesRankingLoss
    },
    "Natural_Questions": {
        "args": {"path": "sentence-transformers/natural-questions"},
        "map_fn": lambda ex: {"anchor": ex["query"], "positive": ex["answer"]},
        "loss": MultipleNegativesRankingLoss
    },
    "PAQ": {
        "args": {"path": "embedding-data/PAQ_pairs"},
        "map_fn": lambda ex: {"anchor": ex["set"][0], "positive": ex["set"][1]},
        "loss": MultipleNegativesRankingLoss
    },
    "Gooaq": {
        "args": {"path": "sentence-transformers/gooaq"},
        "map_fn": lambda ex: {"anchor": ex["question"], "positive": ex["answer"]},
        "loss": MultipleNegativesRankingLoss
    },
    "yahoo_title_answers": {
        "args": {"path": "sentence-transformers/yahoo-answers", "data_dir": "title-answer-pair"},
        "map_fn": lambda ex: {"anchor": ex["title"], "positive": ex["answer"]},
        "loss": MultipleNegativesRankingLoss
    },
    "msmacro_triplet": {
        "args": {"path": "sentence-transformers/msmarco-msmarco-MiniLM-L6-v3", "data_dir": "triplet-hard"},
        "map_fn": lambda ex: {"anchor": ex["query"], "positive": ex["positive"], "negative": ex["negative"]},
        "loss": MultipleNegativesRankingLoss
    },
    "trivia_qa_triplet": {
        "args": {"path": "sentence-transformers/trivia-qa-triplet", "data_dir": "triplet-all"},
        "map_fn": lambda ex: {"anchor": ex["anchor"], "positive": ex["positive"], "negative": ex["negative"]},
        "loss": MultipleNegativesRankingLoss
    },
    "nli_for_simcse_triplet": {
        "args": {"path": "sentence-transformers/nli-for-simcse", "data_dir": "triplet-all"},
        "map_fn": lambda ex: {"anchor": ex["anchor"], "positive": ex["positive"], "negative": ex["negative"]},
        "loss": MultipleNegativesRankingLoss
    },
    "quora_dup_triplet": {
        "args": {"path": "sentence-transformers/quora-duplicates", "data_dir": "triplet-all"},
        "map_fn": lambda ex: {"anchor": ex["anchor"], "positive": ex["positive"], "negative": ex["negative"]},
        "loss": MultipleNegativesRankingLoss
    }
}


if __name__ == "__main__":
    stackexhange_title_best_answers = get_all_data_subset(
        "StackExchange_title_best_answer", 
        "flax-sentence-embeddings/stackexchange_title_best_voted_answer_jsonl", 
        "title_body", "upvoted_answer", MultipleNegativesRankingLoss)
    stackexhange_titlebody_best_answers = get_all_data_subset(
        "StackExchange_titlebody_best_answer",
        "flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl",
        "title_body", "upvoted_answer", MultipleNegativesRankingLoss)
    dataset_names = {**dataset_names, **stackexhange_title_best_answers, **stackexhange_titlebody_best_answers}

    # 1. Load & split each dataset
    dataset_dict = {}
    for name, config in dataset_names.items():
        print("*"*10 + f"Processing: {name}"+ "*"*10)
        out_dir = os.path.join(CACHE_DIR, name)
        if os.path.isdir(out_dir):
            # 1) cached splits already exist → load them
            print(f"Loading cached splits for {name} from {out_dir}")
            splits: DatasetDict = load_from_disk(out_dir)
        else:
            raw = load_dataset(
                **config["args"], 
                trust_remote_code=True,
                # split=f"train[:{NROWS}]"
                )


            if isinstance(raw, DatasetDict):
                # concatenate all splits (train, validation, test, etc.)
                raw = concatenate_datasets(list(raw.values()))

            mapped = raw.map(config["map_fn"], remove_columns=raw.column_names, num_proc=os.cpu_count() // 2)

            # force the correct column order:
            cols = ["anchor", "positive"]
            if "negative" in mapped.column_names:
                cols.append("negative")

            mapped = mapped.select_columns(cols)

            splits = split_dataset(mapped, val_frac=val_frac, test_frac=val_frac, seed=42)
            os.makedirs(out_dir, exist_ok=True)
            splits.save_to_disk(out_dir)
        
        # finally, record it
        dataset_dict[name] = splits


    
    grand_total = 0
    print(f"{'source':40s}  train   val   test   total")
    print("-"*70)
    for name, splits in dataset_dict.items():
        t = len(splits["train"])
        v = len(splits["validation"])
        s = len(splits["test"])
        total = t+v+s
        grand_total += total
        print(f"{name:40s}  {t:6d} {v:6d} {s:6d} {total:8d}")
    print("-"*70)
    print(f"{'Overall':40s}  {grand_total:26d}")
