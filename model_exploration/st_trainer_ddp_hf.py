import argparse
import datetime
import math
import os
import random
import time
from typing import Dict, Union

import distributed
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets
from datasets import config as dataset_config
from datasets import get_dataset_config_names, load_dataset, load_from_disk
from distributed import init_ddp, print0
from dotenv import load_dotenv
from pretokenize import prepare_pre_tokenized_datasets
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
    TripletEvaluator,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import (
    BatchSamplers,
    MultiDatasetBatchSamplers,
    SentenceTransformerTrainingArguments,
)
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import get_scheduler  # For fallback in custom trainer
from utils import (
    PreTokenizedCollator,
    build_dataset_configs_s1,
    build_dataset_configs_s2,
    get_gpu_info,
    load_and_cache_datasets,
    prepare_evaluators,
)

import wandb

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
parser.add_argument("--model_name", type=str, default="nasa-impact/indus-sde-st-v0.1")
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
parser.add_argument("--max_datapoints_per_src_for_eval", type=int, default=20)
parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument(
    "--pretokenize",
    action="store_true",
    help="Enable dataset pretokenization.",
)
parser.add_argument(
    "--no_pretokenize",
    dest="pretokenize",
    action="store_false",
    help="Disable dataset pretokenization.",
)
parser.set_defaults(pretokenize=False)  # Or False, depending on your preferred default

# New arguments for the custom scheduler
parser.add_argument(
    "--custom_lr_scheduler",
    action="store_true",
    help="Enable the custom cosine decay learning rate scheduler.",
)
parser.set_defaults(custom_lr_scheduler=True)
parser.add_argument(
    "--lr_min_custom",
    type=float,
    default=5e-6,
    help="Minimum learning rate for the custom scheduler's linear decay.",
)
parser.add_argument(
    "--cosine_cycle_steps_custom",
    type=int,
    default=8000,
    help="Number of steps for one cosine cycle in the custom scheduler.",
)
parser.add_argument(
    "--cosine_magnitude_fraction_custom",
    type=float,
    default=0.1,
    help="Magnitude of cosine oscillation as a fraction of the current linear LR in the custom scheduler.",
)

parser.add_argument(
    "--lr_scheduler_type",
    type=str,
    default="cosine",
    help="Fallback lr schedular",
)


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
CACHE_DIR = f"../data/stage2_cache/NROWS_{NROWS}"
GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
LEARNING_RATE = args.lr
PRETOKENIZE = args.pretokenize

# Store new custom scheduler args
CUSTOM_LR_SCHEDULER_ENABLED = args.custom_lr_scheduler
LR_MIN_CUSTOM = args.lr_min_custom
COSINE_CYCLE_STEPS_CUSTOM = args.cosine_cycle_steps_custom
COSINE_MAGNITUDE_FRACTION_CUSTOM = args.cosine_magnitude_fraction_custom

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
    "max_datapoints_per_src_for_eval": MAX_DATAPOINTS_PER_SRC_FOR_EVAL,
    "pretokenization": PRETOKENIZE,
    "lr_max (initial_lr)": LEARNING_RATE,
}

if CUSTOM_LR_SCHEDULER_ENABLED:
    wandb_config.update(
        {
            "lr_scheduler_custom_enabled": True,
            "lr_min_custom": LR_MIN_CUSTOM,
            "cosine_cycle_steps_custom": COSINE_CYCLE_STEPS_CUSTOM,
            "cosine_magnitude_fraction_custom": COSINE_MAGNITUDE_FRACTION_CUSTOM,
        },
    )

bf16_supported = torch.cuda.is_bf16_supported()
fp16_supported = torch.cuda.is_available()


# Set dataset config load up to 800gb into the memory for faster training speed
dataset_config.IN_MEMORY_MAX_SIZE = 800 * (1024**3)

# ──────────────── Custom Trainer Class ────────────────
class CustomSentenceTransformerTrainer(SentenceTransformerTrainer):
    def __init__(self, *args_trainer, custom_lr_params: Dict = None, **kwargs_trainer):
        super().__init__(*args_trainer, **kwargs_trainer)
        self.custom_lr_params = custom_lr_params if custom_lr_params is not None else {}

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: torch.optim.Optimizer = None,
    ):
        # self.optimizer should have been created by self.create_optimizer() before this call
        # within self.create_optimizer_and_scheduler().
        if optimizer is None:
            optimizer = self.optimizer
            if optimizer is None:  # Should not happen in normal Trainer flow
                raise ValueError(
                    "Optimizer has not been created yet or not passed to create_scheduler. "
                    "This typically means create_optimizer() was not called before create_scheduler().",
                )

        use_custom_scheduler = self.custom_lr_params.get("enabled_flag", False)

        if use_custom_scheduler:
            lr_min_custom = self.custom_lr_params["lr_min_custom"]
            cosine_cycle_steps_custom = self.custom_lr_params[
                "cosine_cycle_steps_custom"
            ]
            cosine_magnitude_fraction_custom = self.custom_lr_params[
                "cosine_magnitude_fraction_custom"
            ]
            # self.args.learning_rate from TrainingArguments is our lr_max
            optimizer_lr_max = self.args.learning_rate

            # Ensure optimizer's initial_lr is set for LambdaLR factor calculation
            for group in optimizer.param_groups:
                group["initial_lr"] = optimizer_lr_max

            def lr_lambda(current_step: int) -> float:
                num_warmup_steps = self.args.get_warmup_steps(num_training_steps)

                if current_step < num_warmup_steps:
                    if num_warmup_steps == 0:  # Avoid division by zero if no warmup
                        return 1.0  # Factor is 1, LR is optimizer_lr_max
                    return float(current_step) / float(
                        num_warmup_steps,
                    )  # Linear warmup factor
                else:
                    effective_step = current_step - num_warmup_steps
                    total_main_steps = num_training_steps - num_warmup_steps

                    if total_main_steps <= 0:  # No steps after warmup
                        return (
                            lr_min_custom / optimizer_lr_max
                            if optimizer_lr_max > 0
                            else 0.0
                        )

                    # progress_decay: 0 at start of main phase, 1 at end of main phase
                    if (
                        total_main_steps == 1
                    ):  # Single step in the main scheduling phase
                        progress_decay = 1.0
                    else:
                        progress_decay = float(effective_step) / float(
                            total_main_steps - 1,
                        )

                    progress_decay = min(progress_decay, 1.0)  # Clamp progress

                    lr_linear_current = (
                        optimizer_lr_max * (1.0 - progress_decay)
                        + lr_min_custom * progress_decay
                    )

                    amplitude = lr_linear_current * cosine_magnitude_fraction_custom

                    cycle_steps = cosine_cycle_steps_custom
                    cosine_val = 0.0
                    if cycle_steps > 0:
                        cosine_val = math.cos(
                            2 * math.pi * (effective_step % cycle_steps) / cycle_steps,
                        )

                    final_lr_val = lr_linear_current + amplitude * cosine_val
                    final_lr_val = max(0.0, final_lr_val)

                    return (
                        final_lr_val / optimizer_lr_max if optimizer_lr_max > 0 else 0.0
                    )

            # Create and assign the custom scheduler
            self.lr_scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)
        else:
            # If not using custom scheduler, delegate to the parent class's (transformers.Trainer) method.
            # This method will create the scheduler based on self.args.lr_scheduler_type,
            # assign it to self.lr_scheduler, and return it.
            self.lr_scheduler = super().create_scheduler(
                num_training_steps=num_training_steps,
                optimizer=optimizer,
            )

        # Always return self.lr_scheduler, as expected by the base class's create_scheduler signature
        # and some potential call paths.
        return self.lr_scheduler


# ──────────────── Main ────────────────
def main(local_rank, rank):
    global args
    model = SentenceTransformer(
        MODEL_NAME,
        device=f"cuda:{local_rank}",
        tokenizer_kwargs={"model_max_length": MODEL_MAX_LEN, "truncation": True},
        model_kwargs={"torch_dtype": torch.bfloat16 if bf16_supported else None},
    )
    configs = build_dataset_configs_s2(N_DATA_SRC)
    ds_dict = load_and_cache_datasets(configs, CACHE_DIR, NROWS, rank)

    if PRETOKENIZE:
        collator = PreTokenizedCollator(tokenize_fn=model.tokenize)
        train_ds, val_ds, test_ds = prepare_pre_tokenized_datasets(
            ds_dict=ds_dict,
            cache_dir=CACHE_DIR,
            model_name=MODEL_NAME,
            max_len=MODEL_MAX_LEN,
            num_proc=None,
        )
    else:
        collator = None
        train_ds = {n: s["train"] for n, s in ds_dict.items()}
        val_ds = {n: s["validation"] for n, s in ds_dict.items()}
        test_ds = {n: s["test"] for n, s in ds_dict.items()}

    total_rows = (
        sum(len(d) for d in train_ds.values())
        + sum(len(d) for d in val_ds.values())
        + sum(len(d) for d in test_ds.values())
    )

    if distributed.is_main_process():
        wandb_config["train_data_points"] = sum(len(d) for d in train_ds.values())
        wandb_config["val_data_points"] = sum(len(d) for d in val_ds.values())
        wandb_config["test_data_points"] = sum(len(d) for d in test_ds.values())
        wandb_config["total_datapoints"] = total_rows
        wandb.config.update(wandb_config, allow_val_change=True)

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

    print(f"RANK:{rank};Total rows: {total_rows}")

    if distributed.is_main_process():
        eval_strategy = "steps"
        val_evaluator = prepare_evaluators(
            {n: s["validation"] for n, s in ds_dict.items()},
            max_per_split=MAX_DATAPOINTS_PER_SRC_FOR_EVAL,
            BATCH_SIZE=BATCH_SIZE,
        )
    else:
        eval_strategy = "no"
        val_evaluator = None

    # Determine lr_scheduler_type for TrainingArguments
    # If custom scheduler is used, this type is somewhat a placeholder for HF's internal checks,
    # as our create_scheduler override takes precedence. "linear" is a safe default.
    # If custom is not used, this determines the actual scheduler.
    effective_lr_scheduler_type = (
        "linear" if CUSTOM_LR_SCHEDULER_ENABLED else args.lr_scheduler_type
    )  # args.lr_scheduler_type could be "cosine" by default
    if (
        not CUSTOM_LR_SCHEDULER_ENABLED and args.lr_scheduler_type is None
    ):  # Ensure a default if not set and not custom
        effective_lr_scheduler_type = "cosine"

    args = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_ratio=WARMUP_RATIO,
        fp16=not bf16_supported and fp16_supported,
        bf16=bf16_supported,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        # batch_sampler=BatchSamplers.BATCH_SAMPLER,
        # batch_sampler_type=MultiDatasetBatchSamplers.PROPORTIONAL, # TRY THIS
        eval_strategy=eval_strategy,
        eval_steps=EVAL_AND_SAVE_STEPS,
        save_strategy="steps",
        save_steps=EVAL_AND_SAVE_STEPS,
        save_total_limit=2,
        logging_steps=EVAL_AND_SAVE_STEPS,
        learning_rate=LEARNING_RATE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        lr_scheduler_type=effective_lr_scheduler_type,
        report_to="wandb",
        local_rank=local_rank,
        ignore_data_skip=True,
    )

    # Prepare custom_lr_params dictionary to pass to the custom trainer
    custom_lr_params_for_trainer = {
        "enabled_flag": CUSTOM_LR_SCHEDULER_ENABLED,
        "lr_min_custom": LR_MIN_CUSTOM,
        "cosine_cycle_steps_custom": COSINE_CYCLE_STEPS_CUSTOM,
        "cosine_magnitude_fraction_custom": COSINE_MAGNITUDE_FRACTION_CUSTOM,
    }

    trainer = CustomSentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=None,
        loss={n: cfg["loss"](model) for n, cfg in configs.items()},
        evaluator=val_evaluator,
        data_collator=collator,
        custom_lr_params=custom_lr_params_for_trainer,  # Pass our custom params
    )

    if RESUME_CHECKPOINT_PATH is not None:
        print(f"RANK:{rank};Resuming training...")
        trainer.train(resume_from_checkpoint=RESUME_CHECKPOINT_PATH)
    else:
        print(f"RANK:{rank};Starting training...")
        trainer.train()

    print(f"RANK:{rank};Finished training...")

    if distributed.is_main_process():
        test_evaluator = prepare_evaluators(
            {n: s["test"] for n, s in ds_dict.items()},
            max_per_split=None,
            BATCH_SIZE=BATCH_SIZE,
        )
        model_to_eval = (
            model.module if isinstance(model, DistributedDataParallel) else model
        )
        model_to_eval.to(f"cuda:{local_rank}")  # Keep it on rank 0's GPU
        print("Started Test Evaluation ...")
        test_results = test_evaluator(model_to_eval)
        wandb.log({f"Test Evaluation": test_results})
        print("Finished Test Evaluation on Rank 0.")
        print(test_results)

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
                entity="impact-ibm-collaboration",
                project="nasa-indus-sde-s2",
                mode=WB_MODE,
                id=RESUME_RUN_ID,
                resume="must",
                # group="ddp_run",
                # job_type="train",
                # reinit=False,
            )
        else:
            wandb.init(
                entity="impact-ibm-collaboration",
                project="nasa-indus-sde-s2",
                mode=WB_MODE,
                # group="ddp_run",
                # job_type="train",
                # reinit=False,
            )

        wandb_config = {**wandb_config, **get_gpu_info()}

    torch.distributed.barrier(device_ids=[local_rank])

    start_time = time.time()
    main(local_rank, rank)

    if distributed.is_main_process():
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"RANK: {rank}; Elapsed time: {elapsed_time:.2f} seconds")
        wandb.log({"timing/overall_script_seconds": elapsed_time})
        hours = elapsed_time / 3600
        wandb.log({"timing/overall_script_hours": hours})
        wandb.finish()

    torch.distributed.destroy_process_group()
