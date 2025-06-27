# Sentence Transformer Training Workflow

This repository provides a comprehensive workflow for fine-tuning Sentence Transformer models using PyTorch, Hugging Face `datasets`, and `transformers`. It is designed for multi-GPU training using `torchrun` and supports features like dataset pre-tokenization, Weights & Biases logging, and custom learning rate schedulers.

## 1. Prerequisites

Before you begin, ensure you have the following installed and configured:

* **Git**: To clone the repository.
* **Python 3.8+**: The programming language used.
* **uv**: A fast Python package installer and resolver, used for setting up the project environment. You can install it with `pip install uv`.
* **Kaggle Account & API Token**: Required for downloading the `arxiv` dataset. Make sure you have your `kaggle.json` file set up.
* **Hugging Face Account & Token**: Required for accessing models and datasets from the Hugging Face Hub, especially private ones.
* **Weights & Biases Account & API Key**: For logging training metrics and model checkpoints.

## 2. Setup and Installation

Follow these steps to set up your project environment.

### Step 2.1: Clone the Repository

First, clone the project from GitHub:

```bash
git clone https://github.com/NASA-IMPACT/st-training-workflow
cd st-training-workflow
```

### Step 2.2: Set Up the Python Environment

This project uses `uv` to manage dependencies. To create the virtual environment and install the required packages from `requirements.txt`, run the following command from the root of the repository:

```bash
uv venv
uv pip install -r requirements.txt
```
Activate the environment with:
```bash
source .venv/bin/activate
```

## 3. Data Preparation

Most datasets are downloaded automatically from the Hugging Face Hub during the training process. However, one of the datasets used in Stage 2 training (`arxiv`) must be downloaded manually from Kaggle.

### Step 3.1: Create Data Directories

From the root of the repository, create the necessary directories for the raw data. The training script expects the data to be in a `data_prep` directory located *outside* the `model_exploration` folder.

```bash
mkdir -p data_prep/raw/
```

### Step 3.2: Download and Extract the ArXiv Dataset

Use the Kaggle API to download the dataset and unzip it into the `data_prep/raw/` directory.

```bash
# Note: Ensure your Kaggle API token is configured correctly
curl -L -o data_prep/raw/arxiv.zip https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv

# Unzip the contents into the raw data directory
unzip data_prep/raw/arxiv.zip -d data_prep/raw/
```
This will place the `arxiv-metadata-oai-snapshot.json` file where the script `model_exploration/utils.py` expects to find it.

## 4. Environment Configuration

To manage secrets and important configuration, create a `.env` file inside the `model_exploration` directory. This is where you will store your API keys and other environment variables.

### Step 4.1: Navigate to the `model_exploration` Directory

```bash
cd model_exploration
```

### Step 4.2: Create the `.env` File

Create a file named `.env` and add the following variables.

```env
# .env file in the 'model_exploration' directory

# W&B: Set to "end" to upload the final model checkpoint as a W&B artifact.
export WANDB_LOG_MODEL="end"

# W&B: Your Weights & Biases API key for logging.
export WANDB_API_KEY="YOUR_WANDB_API_KEY"

# Hugging Face: Your token for accessing models/datasets from the HF Hub.
export HUGGINGFACE_TOKEN="YOUR_HUGGINGFACE_TOKEN"
```

Replace `"YOUR_WANDB_API_KEY"` and `"YOUR_HUGGINGFACE_TOKEN"` with your actual credentials.

## 5. Running the Training

The training is initiated using `torchrun` for distributed data parallel (DDP) training across multiple GPUs.

### Step 5.1: Start the Training Script

Ensure you are in the `model_exploration` directory before running the command.

```bash
# Example training command
torchrun --nproc_per_node=auto st_trainer_ddp_hf.py \
    --model_name "nasa-impact/indus-sde-st-v0.1" \
    --num_train_epochs 1 \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --lr 2e-5 \
    --eval_and_save_steps 1000 \
    --output_base "../training_output"
```

### Step 5.2: Command-Line Arguments

You can customize the training run using various command-line arguments. Here are some of the key options available in `st_trainer_ddp_hf.py`:

| Argument                          | Type    | Default                             | Description                                                                    |
| --------------------------------- | ------- | ----------------------------------- | ------------------------------------------------------------------------------ |
| `--nrows`                         | int     | `None`                              | Number of rows to use from each dataset (for quick testing).                   |
| `--n_data_src`                    | int     | `None`                              | Limit the number of data sources for training.                                 |
| `--model_name`                    | str     | `nasa-impact/indus-sde-st-v0.1`     | The base Sentence Transformer model to fine-tune from the Hugging Face Hub.    |
| `--output_base`                   | str     | `tmp_models`                        | The base directory where training outputs and checkpoints will be saved.       |
| `--num_train_epochs`              | int     | `1`                                 | The total number of training epochs to perform.                                |
| `--batch_size`                    | int     | `64`                                | The batch size per device (GPU) for training.                                  |
| `--lr`                            | float   | `2e-5`                              | The initial learning rate for the AdamW optimizer.                             |
| `--warmup_ratio`                  | float   | `0.1`                               | The proportion of training steps for the learning rate warm-up.                |
| `--eval_and_save_steps`           | int     | `1000`                              | The frequency (in steps) to run evaluation and save a model checkpoint.        |
| `--gradient_accumulation_steps`   | int     | `8`                                 | Number of steps to accumulate gradients before performing an optimizer step.   |
| `--pretokenize`                   | flag    | `False`                             | Enable dataset pre-tokenization to speed up training by caching tokenized data.  |
| `--resume_checkpoint_path`        | str     | `None`                              | Path to a checkpoint to resume training from.                                  |
| `--resume_run_id`                 | str     | `None`                              | The Weights & Biases run ID to resume logging to.                              |
| `--custom_lr_scheduler`           | flag    | `True`                              | Use the custom learning rate scheduler with cosine decay.                      |


## 6. Codebase Overview

* **`st_trainer_ddp_hf.py`**: The main entry point for the training script. It handles argument parsing, DDP setup, data loading, trainer initialization, and evaluation.
* **`utils.py`**: Contains helper functions for building dataset configurations (`build_dataset_configs_s1`, `build_dataset_configs_s2`), loading and caching datasets (`load_and_cache_datasets`), and preparing evaluation suites (`prepare_evaluators`).
* **`distributed.py`**: Provides utilities for setting up and managing the distributed training environment.
* **`pretokenize.py`**: Contains the logic for pre-tokenizing the datasets and caching them to disk to accelerate subsequent training runs.
* **`requirements.txt`**: A list of Python packages required to run the code.
