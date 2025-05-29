import os
import math
import argparse
import datetime
import random
from typing import Union
from datasets import (
    load_dataset,
    load_from_disk,
    Dataset,
    DatasetDict,
    concatenate_datasets,
    get_dataset_config_names,
    IterableDataset,
)
import wandb
import torch
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
# from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments, BatchSamplers
from sentence_transformers.evaluation import InformationRetrievalEvaluator, TripletEvaluator, SequentialEvaluator
from dotenv import load_dotenv
import distributed
from distributed import init_ddp, print0
from torch.nn.parallel import DistributedDataParallel

from utils import build_dataset_configs, load_and_cache_datasets, prepare_evaluators, get_gpu_info, MultipleNegativesRankingLoss
# ──────────────── Constants ────────────────

parser = argparse.ArgumentParser(description="Sentence Transformer Training Config")

parser.add_argument("--nrows", type=int, default=None)
parser.add_argument("--n_data_src", type=int, default=None, help="number of data sources to use for training")
parser.add_argument("--val_frac", type=float, default=0.05)
parser.add_argument("--test_frac", type=float, default=0.05)
parser.add_argument("--model_max_len", type=int, default=1024)
parser.add_argument("--model_name", type=str, default="nasa-impact/indus-sde-v0.2")
parser.add_argument("--output_base", type=str, default="tmp_models")
parser.add_argument("--wb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
parser.add_argument("--resume_checkpoint_path", type=str, default=None)
parser.add_argument("--resume_run_id", type=str, default=None)
parser.add_argument("--num_train_epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--eval_and_save_steps", type=int, default=1000)
parser.add_argument("--max_datapoints_per_src_for_eval", type=int, default=20)
parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--streaming", action="store_true", help="Enable dataset streaming.")
parser.add_argument("--no_streaming", dest="streaming", action="store_false", help="Disable dataset streaming.")
parser.set_defaults(streaming=True) # Or False, depending on your preferred default



args = parser.parse_args()

NROWS      = args.nrows
VAL_FRAC   = args.val_frac
TEST_FRAC  = args.test_frac
MODEL_MAX_LEN = args.model_max_len
MODEL_NAME = args.model_name
OUTPUT_BASE= args.output_base
WB_MODE    = args.wb_mode
RESUME_CHECKPOINT_PATH = args.resume_checkpoint_path
RESUME_RUN_ID = args.resume_run_id
NUM_TRAIN_EPOCHS = args.num_train_epochs
BATCH_SIZE = args.batch_size
WARMUP_RATIO = args.warmup_ratio
EVAL_AND_SAVE_STEPS = args.eval_and_save_steps
MAX_DATAPOINTS_PER_SRC_FOR_EVAL = args.max_datapoints_per_src_for_eval
N_DATA_SRC = args.n_data_src
CACHE_DIR = f"./data/stage1_cache/NROWS_{NROWS}"
GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
LEARNING_RATE = args.lr
STREAMING = args.streaming

load_dotenv()
current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y%m%d_%H-%M-%S")
os.makedirs(CACHE_DIR, exist_ok=True)
assert os.getenv("WANDB_LOG_MODEL") == "end"

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
    "max_datapoints_per_src_for_eval": MAX_DATAPOINTS_PER_SRC_FOR_EVAL
}
bf16_supported = torch.cuda.is_bf16_supported()
fp16_supported = torch.cuda.is_available()


def compute_max_steps(n_samples):
    if STREAMING:
        world_size = int(os.environ["WORLD_SIZE"])
        total_train_samples_one_epoch = sum([nsize.get("train", 0) for src_name, nsize in n_samples.items()])
        effective_global_batch_size_for_update = BATCH_SIZE * world_size * GRADIENT_ACCUMULATION_STEPS
        steps_per_epoch = math.ceil(total_train_samples_one_epoch / effective_global_batch_size_for_update)
        max_steps = int(steps_per_epoch * NUM_TRAIN_EPOCHS)
    else:
        max_steps = -1

    return max_steps

# ──────────────── Main ────────────────
def main(local_rank, rank):
    model   = SentenceTransformer(MODEL_NAME, tokenizer_kwargs={"model_max_length": MODEL_MAX_LEN})
    model.to(f"cuda:{local_rank}")
    configs = build_dataset_configs(N_DATA_SRC)
    train_ds, other_ds, n_samples = load_and_cache_datasets(
        configs, 
        CACHE_DIR, 
        NROWS, 
        rank, 
        streaming=STREAMING, 
        val_frac=VAL_FRAC, 
        test_frac=TEST_FRAC
    )

    # train_ds = {n: s["train"]      for n, s in ds_dict.items()}
    val_ds   = {n: s["validation"] for n, s in other_ds.items()}
    test_ds = {n: s["test"] for n, s in other_ds.items()}

    if distributed.is_main_process():
        wandb_config["dataset_src_sizes"] = n_samples
        wandb.config.update(wandb_config, allow_val_change=True)


    if RESUME_CHECKPOINT_PATH is not None:
        output_dir = "/".join(RESUME_CHECKPOINT_PATH.split("/")[:-2])
    else:
        output_dir = str(os.path.join(OUTPUT_BASE, f"nrows_{NROWS}__nsrc_{N_DATA_SRC}", f"timestamp_{formatted_datetime}" ,MODEL_NAME.split("/")[-1]))
    os.makedirs(output_dir, exist_ok=True)

    # print(f"RANK:{rank};Total rows: {total_rows}")

    if distributed.is_main_process():
        eval_strategy = "steps"
        val_evaluator = prepare_evaluators(val_ds, max_per_split=MAX_DATAPOINTS_PER_SRC_FOR_EVAL, BATCH_SIZE=BATCH_SIZE)
    else:
        eval_strategy = "no"
        val_evaluator = None

    args = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_ratio=WARMUP_RATIO,
        fp16=not bf16_supported and fp16_supported,
        bf16=bf16_supported,
        # batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy=eval_strategy,
        eval_steps=EVAL_AND_SAVE_STEPS,
        save_strategy="steps",
        save_steps=EVAL_AND_SAVE_STEPS,
        save_total_limit=2,
        logging_steps=EVAL_AND_SAVE_STEPS,
        learning_rate=LEARNING_RATE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        lr_scheduler_type="cosine",
        report_to="wandb",
        local_rank=local_rank,
        max_steps = compute_max_steps(n_samples),
        accelerator_config={'dispatch_batches': False},
        # split_batches=True,
    )
    

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=None,
        # loss={n: cfg["loss"](model) for n, cfg in configs.items()},
        loss = MultipleNegativesRankingLoss(model),
        evaluator=val_evaluator,
    )

    if RESUME_CHECKPOINT_PATH is not None:
        print(f"RANK:{rank};Resuming training...")
        trainer.train(resume_from_checkpoint=RESUME_CHECKPOINT_PATH)
    else:
        print(f"RANK:{rank};Starting training...")
        trainer.train()

    print(f"RANK:{rank};Finished training...")

    
    if distributed.is_main_process():
        test_evaluator = prepare_evaluators(test_ds, max_per_split=None, BATCH_SIZE=BATCH_SIZE)
        model_to_eval = model.module if isinstance(model, DistributedDataParallel) else model
        model_to_eval.to(f"cuda:{local_rank}") # Keep it on rank 0's GPU
        print("Started Test Evaluation ...")
        test_results = test_evaluator(model_to_eval)
        wandb.log({f"Test Evaluation": test_results})
        print("Finished Test Evaluation on Rank 0.")

    torch.distributed.barrier(device_ids=[local_rank])

    # model.save_pretrained(os.path.join(output_dir, "final_name"))


if __name__ == "__main__":
    run = None
    local_rank, rank = init_ddp(True)
    print(f"RANK: {rank}; LOCAL_RANK: {local_rank}.")

    if distributed.is_main_process():
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        if RESUME_RUN_ID is not None:
            assert RESUME_RUN_ID is not None
            wandb.init(
                project="nasa_st_traning",
                mode=WB_MODE,
                id=RESUME_RUN_ID,
                resume="must",
                # group="ddp_run",
                # job_type="train",
                # reinit=False,
            )
        else:
            wandb.init(
                project="nasa_st_traning", 
                mode=WB_MODE,
                # group="ddp_run",
                # job_type="train",
                # reinit=False,
                )

        wandb_config = {**wandb_config, **get_gpu_info()}

    torch.distributed.barrier(device_ids=[local_rank])

    main(local_rank, rank)

    if distributed.is_main_process():
        wandb.finish()

    torch.distributed.destroy_process_group()

