import os
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer
import torch.distributed as dist

def prepare_pre_tokenized_datasets(
    ds_dict: dict[str, DatasetDict],
    cache_dir: str,
    model_name: str,
    max_len: int,
    num_proc: int | None = None,
):
    """
    Given in‐memory ds_dict[name] = DatasetDict({train/validation/test}),
    tokenizes each split (if not already cached under cache_dir/<name>_tok),
    and returns three dicts mapping name->datasets.Dataset (train/val/test).
    """
    is_ddp = dist.is_available() and dist.is_initialized()
    rank   = dist.get_rank() if is_ddp else 0

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=max_len,
        truncation=True
    )

    train_ds, val_ds, test_ds = {}, {}, {}

    for name, splits in ds_dict.items():
        tok_dir = os.path.join(cache_dir, f"{name}_tok")

        # 1) if already tokenized, load from disk
        if os.path.isdir(tok_dir):
            if is_ddp:
                dist.barrier()
            tok_splits = load_from_disk(tok_dir)

        # 2) else rank 0 does the mapping + save, others wait & load
        else:
            if rank == 0:
                tok_splits = DatasetDict()
                def tok_fn(ex):
                    out = {}
                    a = tokenizer(ex["anchor"],  max_length=max_len, padding="max_length", truncation=True)
                    p = tokenizer(ex["positive"], max_length=max_len, padding="max_length", truncation=True)
                    out["anchor_input_ids"]      = a["input_ids"]
                    out["anchor_attention_mask"] = a["attention_mask"]
                    out["positive_input_ids"]      = p["input_ids"]
                    out["positive_attention_mask"] = p["attention_mask"]
                    if "negative" in ex:
                        n = tokenizer(ex["negative"], max_length=max_len, padding="max_length", truncation=True)
                        out["negative_input_ids"]      = n["input_ids"]
                        out["negative_attention_mask"] = n["attention_mask"]
                    return out

                for split_name, ds in splits.items():
                    tok_ds = ds.map(
                        tok_fn,
                        remove_columns=ds.column_names,
                        num_proc=num_proc or max(1, os.cpu_count() // 2),
                    )
                    tok_splits[split_name] = tok_ds

                os.makedirs(tok_dir, exist_ok=True)
                tok_splits.save_to_disk(tok_dir)

                if is_ddp:
                    dist.barrier()
            else:
                if is_ddp:
                    dist.barrier()
                tok_splits = load_from_disk(tok_dir)

        # 3) **Return the raw HF Dataset** (no PyTorch wrapper)
        train_ds[name] = tok_splits["train"]
        val_ds[name] = tok_splits["validation"]
        test_ds[name] = tok_splits["test"]

    return train_ds, val_ds, test_ds