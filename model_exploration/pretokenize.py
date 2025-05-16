import os
from datasets import load_dataset, load_from_disk, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer
from utils import split_dataset, PreTokenizedPyTorchDataset

def prepare_pre_tokenized_splits(
    name: str,
    cfg: dict,
    cache_dir: str,
    model_name: str,
    max_len: int,
    nrows: int = None,
    num_proc: int | None = None
) -> DatasetDict:
    """
    Ensure that <cache_dir>/<name>_tok exists; if not,:
      1. Load and map raw splits via cfg["args"] & cfg["map_fn"].
      2. Split into train/validation/test.
      3. Tokenize anchor/positive[/negative] fields.
      4. Save tokenized splits under <cache_dir>/<name>_tok.

    Returns a HF DatasetDict with keys 'train', 'validation', 'test'.
    """
    tok_dir = os.path.join(cache_dir, f"{name}_tok")
    if os.path.isdir(tok_dir):
        return load_from_disk(tok_dir)

    # Load or build original mapped splits
    orig_dir = os.path.join(cache_dir, name)
    if os.path.isdir(orig_dir):
        splits: DatasetDict = load_from_disk(orig_dir)
    else:
        # 1) Load raw dataset
        if nrows:
            raw = load_dataset(**cfg["args"], split=f"train[:{nrows}]", trust_remote_code=True)
        else:
            raw = load_dataset(**cfg["args"], trust_remote_code=True)

        # 2) Merge and map
        if isinstance(raw, DatasetDict):
            raw = concatenate_datasets(list(raw.values()))
        mapped = raw.map(
            cfg["map_fn"],
            remove_columns=raw.column_names,
            num_proc=num_proc or max(1, os.cpu_count() // 2)
        )

        # 3) Split
        splits = split_dataset(mapped)
        os.makedirs(orig_dir, exist_ok=True)
        splits.save_to_disk(orig_dir)

    # 4) Tokenize splits
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=max_len,
        truncation=True
    )

    def tok_fn(ex):
        out = {}
        a = tokenizer(ex["anchor"], max_length=max_len, truncation=True)
        out["anchor_input_ids"]      = a["input_ids"]
        out["anchor_attention_mask"] = a["attention_mask"]

        p = tokenizer(ex["positive"], max_length=max_len, truncation=True)
        out["positive_input_ids"]      = p["input_ids"]
        out["positive_attention_mask"] = p["attention_mask"]

        if "negative" in ex:
            n = tokenizer(ex["negative"], max_length=max_len, truncation=True)
            out["negative_input_ids"]      = n["input_ids"]
            out["negative_attention_mask"] = n["attention_mask"]
        return out

    tok_splits = splits.map(
        tok_fn,
        remove_columns=splits.column_names,
        num_proc=num_proc or max(1, os.cpu_count() // 2)
    )

    os.makedirs(tok_dir, exist_ok=True)
    tok_splits.save_to_disk(tok_dir)
    return tok_splits


def prepare_pre_tokenized_datasets(
    configs: dict,
    cache_dir: str,
    model_name: str,
    max_len: int,
    nrows: int = None,
    num_proc: int = None
):
    """
    For each (name, cfg) in configs:
      - ensure tokenized splits via prepare_pre_tokenized_splits
      - wrap split into PreTokenizedPyTorchDataset

    Returns three dicts: train_ds, val_ds, test_ds mapping names to PyTorch datasets.
    """
    train_ds, val_ds, test_ds = {}, {}, {}
    for name, cfg in configs.items():
        tok_splits = prepare_pre_tokenized_splits(
            name=name,
            cfg=cfg,
            cache_dir=cache_dir,
            model_name=model_name,
            max_len=max_len,
            nrows=nrows,
            num_proc=num_proc
        )
        train_ds[name] = PreTokenizedPyTorchDataset(tok_splits["train"])
        val_ds[name]   = PreTokenizedPyTorchDataset(tok_splits["validation"])
        test_ds[name]  = PreTokenizedPyTorchDataset(tok_splits["test"])
    return train_ds, val_ds, test_ds
