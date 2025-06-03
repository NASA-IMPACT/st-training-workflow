import argparse
import datetime
import os
import random
from typing import Union

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
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
    TripletEvaluator,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import (
    BatchSamplers,
    SentenceTransformerTrainingArguments,
)
from utils import (
    build_dataset_configs,
    get_gpu_info,
    load_and_cache_datasets,
    prepare_evaluators,
)

# ──────────────── Constants ────────────────

parser = argparse.ArgumentParser(description="Sentence Transformer Training Config")

parser.add_argument("--nrows", type=int, default=None)
parser.add_argument(
    "--n_data_src",
    type=int,
    default=None,
    help="number of data sources to use for training",
)
parser.add_argument("--val_frac", type=float, default=0.05)
parser.add_argument("--test_frac", type=float, default=0.05)
parser.add_argument("--model_max_len", type=int, default=1024)
parser.add_argument("--model_name", type=str, default="nasa-impact/indus-sde-v0.2")
parser.add_argument("--output_base", type=str, default="tmp_models")
parser.add_argument(
    "--wb_mode",
    type=str,
    default="online",
    choices=["online", "offline", "disabled"],
)
parser.add_argument("--resume_checkpoint_path", type=str, default=None)
parser.add_argument("--resume_run_id", type=str, default=None)
parser.add_argument("--num_train_epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--eval_and_save_steps", type=int, default=1000)
parser.add_argument("--max_datapoints_per_src_for_eval", type=int, default=2)
parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-5)


args = parser.parse_args()

NROWS = args.nrows
VAL_FRAC = args.val_frac
TEST_FRAC = args.test_frac
MODEL_MAX_LEN = args.model_max_len
MODEL_NAME = args.model_name
OUTPUT_BASE = args.output_base
WB_MODE = args.wb_mode
RESUME_CHECKPOINT_PATH = args.resume_checkpoint_path
RESUME_RUN_ID = args.resume_run_id
NUM_TRAIN_EPOCHS = args.num_train_epochs
BATCH_SIZE = args.batch_size
WARMUP_RATIO = args.warmup_ratio
EVAL_AND_SAVE_STEPS = args.eval_and_save_steps
MAX_DATAPOINTS_PER_SRC_FOR_EVAL = args.max_datapoints_per_src_for_eval
N_DATA_SRC = args.n_data_src
CACHE_DIR = f"../data/stage1_cache/NROWS_{NROWS}"
GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
LEARNING_RATE = args.lr

load_dotenv()
current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y%m%d_%H-%M-%S")
os.makedirs(CACHE_DIR, exist_ok=True)
assert os.getenv("WANDB_LOG_MODEL") == "end"
wandb.login(key=os.getenv("WANDB_API_KEY"))
if RESUME_RUN_ID is not None:
    assert RESUME_RUN_ID is not None
    wandb.init(
        project="nasa_st_traning",
        mode=WB_MODE,
        id=RESUME_RUN_ID,
        resume="must",
    )
else:
    wandb.init(project="nasa_st_traning", mode=WB_MODE)

wandb_config = {
    "model_name": MODEL_NAME,
    "nrows": NROWS,
    "val_frac": VAL_FRAC,
    "test_frac": TEST_FRAC,
    "model_max_len": MODEL_MAX_LEN,
    "data_cache_dir": CACHE_DIR,
    "num_train_epochs": NUM_TRAIN_EPOCHS,
    "batch_size": BATCH_SIZE,
    "warmup_ratio": WARMUP_RATIO,
    "eval_and_save_steps": EVAL_AND_SAVE_STEPS,
    "max_datapoints_per_src_for_eval": MAX_DATAPOINTS_PER_SRC_FOR_EVAL,
}
bf16_supported = torch.cuda.is_bf16_supported()
fp16_supported = torch.cuda.is_available()


# ──────────────── Main ────────────────
def main():
    model = SentenceTransformer(
        MODEL_NAME,
        tokenizer_kwargs={"model_max_length": MODEL_MAX_LEN},
    )
    configs = build_dataset_configs(N_DATA_SRC)
    ds_dict = load_and_cache_datasets(configs, CACHE_DIR, NROWS)

    train_ds = {n: s["train"] for n, s in ds_dict.items()}
    val_ds = {n: s["validation"] for n, s in ds_dict.items()}
    test_ds = {n: s["test"] for n, s in ds_dict.items()}

    total_rows = (
        sum(d.num_rows for d in train_ds.values())
        + sum(d.num_rows for d in val_ds.values())
        + sum(d.num_rows for d in test_ds.values())
    )
    wandb_config["train_data_points"] = sum(d.num_rows for d in train_ds.values())
    wandb_config["val_data_points"] = sum(d.num_rows for d in val_ds.values())
    wandb_config["test_data_points"] = sum(d.num_rows for d in test_ds.values())
    wandb_config["total_datapoints"] = total_rows
    wandb.config.update(wandb_config)

    if RESUME_CHECKPOINT_PATH is not None:
        output_dir = "/".join(RESUME_CHECKPOINT_PATH.split("/")[:-2])
    else:
        output_dir = str(
            os.path.join(
                OUTPUT_BASE,
                f"nrows_{NROWS}__nsrc_{N_DATA_SRC}",
                f"timestamp_{formatted_datetime}",
                MODEL_NAME.split("/")[-1],
            ),
        )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Total rows: {total_rows}")

    args = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_ratio=WARMUP_RATIO,
        fp16=not bf16_supported and fp16_supported,
        bf16=bf16_supported,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=EVAL_AND_SAVE_STEPS,
        save_strategy="steps",
        save_steps=EVAL_AND_SAVE_STEPS,
        save_total_limit=2,
        logging_steps=EVAL_AND_SAVE_STEPS,
        learning_rate=LEARNING_RATE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        lr_scheduler_type="cosine",
        report_to="wandb",
    )

    if MAX_DATAPOINTS_PER_SRC_FOR_EVAL:
        val_subset = {
            k: v.select(range(min(MAX_DATAPOINTS_PER_SRC_FOR_EVAL, len(v))))
            for k, v in val_ds.items()
        }
    else:
        val_subset = val_ds

    val_evaluator = prepare_evaluators(
        val_ds,
        max_per_split=MAX_DATAPOINTS_PER_SRC_FOR_EVAL,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_subset,
        loss={n: cfg["loss"](model) for n, cfg in configs.items()},
        evaluator=val_evaluator,
    )

    if RESUME_CHECKPOINT_PATH is not None:
        print("Resuming training...")
        trainer.train(resume_from_checkpoint=RESUME_CHECKPOINT_PATH)
    else:
        print("Starting training...")
        trainer.train()

    print("Finished training...")

    test_evaluator = prepare_evaluators(test_ds, max_per_split=None)
    print("Started Test Evaluation ...")
    test_results = test_evaluator(model)
    wandb.log({f"Test Evaluation": test_results})

    # model.save_pretrained(os.path.join(output_dir, "final_name"))


if __name__ == "__main__":
    wandb_config = {**wandb_config, **get_gpu_info()}
    main()

wandb.finish()
