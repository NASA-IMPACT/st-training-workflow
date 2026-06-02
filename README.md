# Sentence Transformer Training Workflow

This repository provides a comprehensive workflow for fine-tuning Sentence Transformer models using PyTorch, Hugging Face `datasets`, and `transformers`. It is designed for multi-GPU training using `torchrun` and supports features like dataset pre-tokenization, Weights & Biases logging, and custom learning rate schedulers.

---

## INDUS-SDE-ST — Sentence Transformer for Scientific Content Discovery

These workflows produce **INDUS-SDE-ST**, the semantic-discovery model from our paper:

> **INDUS-SDE: A Language Model for Scientific Content Curation and Discovery**
> Pantha et al. — **KDD 2026, AI for Sciences Track** · DOI: [10.1145/3770855.3818847](https://doi.org/10.1145/3770855.3818847)

**INDUS-SDE** is a domain-adapted encoder pretrained with Weighted Dynamic Masking on NASA's Science Discovery Engine (SDE) corpus (pretraining code: [`NASA-IMPACT/mlm-fine-tuning`](https://github.com/NASA-IMPACT/mlm-fine-tuning)). **INDUS-SDE-ST** — built here — fine-tunes that encoder into a sentence transformer for semantic retrieval over heterogeneous, web-sourced scientific content.

### Models & data
| Artifact | Hugging Face |
|---|---|
| Sentence transformer | [`nasa-impact/indus-sde-st-v0.2`](https://huggingface.co/nasa-impact/indus-sde-st-v0.2) |
| Binary (EQAT) embeddings | [`nasa-impact/indus-sde-st-equat-v0.1`](https://huggingface.co/nasa-impact/indus-sde-st-equat-v0.1) |
| Base encoder (INDUS-SDE) | [`nasa-impact/indus-sde-v0.2`](https://huggingface.co/nasa-impact/indus-sde-v0.2) |
| NASA SDE IR benchmark | [`nasa-impact/nasa-sde-IR-benchmark-20251024-v5`](https://huggingface.co/datasets/nasa-impact/nasa-sde-IR-benchmark-20251024-v5) |
| Pretraining (MLM / WDM) code | [`NASA-IMPACT/mlm-fine-tuning`](https://github.com/NASA-IMPACT/mlm-fine-tuning) |

### Where the paper's data pipeline lives
- Stage-2 synthetic pair generation (Instructor schema + system prompt) → [`gen_data_stage2/`](gen_data_stage2/) · [module README](gen_data_stage2/README.md)
- SDE content-relevancy filtering (Pydantic AI) → [`gen_data_stage2/filter2_llm_based/`](gen_data_stage2/filter2_llm_based/)
- Stage-2 data prep / CMR–PDS pairs → [`data_prep/`](data_prep/)
- Benchmark & per-checkpoint eval dumps → [`eval/results_json/`](eval/results_json/)

### NASA SDE IR — headline result (vs. released models)

On the in-domain **NASA SDE IR benchmark** ([`nasa-impact/nasa-sde-IR-benchmark-20251024-v5`](https://huggingface.co/datasets/nasa-impact/nasa-sde-IR-benchmark-20251024-v5)), INDUS-SDE-ST is the strongest retriever among released models (the comparison reported in the paper):

| Model | MRR@1 | MRR@5 | NDCG@1 | NDCG@5 |
|---|---|---|---|---|
| INDUS-ST | 0.1616 | 0.2131 | 0.1616 | 0.2351 |
| ModernBERT-ST | 0.1765 | 0.2137 | 0.1765 | 0.2288 |
| **INDUS-SDE-ST** | **0.2343** | **0.3122** | **0.2343** | **0.3445** |
| OpenAI text-embedding-3-small | 0.2034 | 0.2619 | 0.2034 | 0.2870 |

> **Note:** INDUS-SDE-ST is the best **in-domain** retriever among released models (above). The Stage-2 ablation below compares *internal* training checkpoints; one unreleased run (`dutiful-thunder-110`) scores slightly higher on NASA SDE IR by pushing in-domain harder, but it trades away NanoBEIR and NASA SMD IR — so the **balanced** checkpoint (INDUS-SDE-ST) was the one released.

### Stage-2 checkpoint ablation (extended results)

Full run-by-run Stage-2 evaluation of internal checkpoints across all three benchmarks.

<table>
<thead>
<tr>
  <th rowspan="3">SN</th><th rowspan="3">Checkpoint</th><th rowspan="3">Training approach</th>
  <th colspan="4">NanoBEIR</th><th colspan="4">NASA SMD IR</th><th colspan="4">NASA SDE IR</th>
</tr>
<tr>
  <th colspan="2">MRR</th><th colspan="2">NDCG</th>
  <th colspan="2">MRR</th><th colspan="2">NDCG</th>
  <th colspan="2">MRR</th><th colspan="2">NDCG</th>
</tr>
<tr>
  <th>@1</th><th>@5</th><th>@1</th><th>@5</th>
  <th>@1</th><th>@5</th><th>@1</th><th>@5</th>
  <th>@1</th><th>@5</th><th>@1</th><th>@5</th>
</tr>
</thead>
<tbody>
<tr><td>1</td><td>INDUS-ST</td><td>Baseline</td><td>0.50</td><td><b>0.61</b></td><td>0.50</td><td>0.54</td><td><b>0.53</b></td><td><b>0.58</b></td><td><b>0.53</b></td><td><b>0.61</b></td><td>0.16</td><td>0.21</td><td>0.16</td><td>0.23</td></tr>
<tr><td>2</td><td>indus-sde-st-v0.1</td><td>Stage 1 ST</td><td><b>0.52</b></td><td><b>0.61</b></td><td><b>0.52</b></td><td><b>0.55</b></td><td>0.47</td><td>0.53</td><td>0.47</td><td>0.55</td><td><b>0.19</b></td><td><b>0.24</b></td><td><b>0.19</b></td><td><b>0.26</b></td></tr>
<tr><td colspan="15"><em>Stage 2 Training Experiments (FP32)</em></td></tr>
<tr><td>3</td><td>whole-moon-14</td><td>Trained on full Stage 2 data</td><td>0.40</td><td>0.51</td><td>0.40</td><td>0.45</td><td>0.38</td><td>0.45</td><td>0.38</td><td>0.47</td><td>0.17</td><td>0.23</td><td>0.17</td><td>0.26</td></tr>
<tr><td>4</td><td>atomic-plasma-15</td><td>Stage 2 subset (NASA-SDE + ADS)</td><td><b>0.54</b></td><td><b>0.63</b></td><td><b>0.54</b></td><td><b>0.56</b></td><td><b>0.48</b></td><td><b>0.54</b></td><td><b>0.48</b></td><td><b>0.57</b></td><td>0.19</td><td>0.25</td><td>0.19</td><td>0.28</td></tr>
<tr><td>5</td><td>peach-night-57</td><td>Stage 2 (No Cite) + weighted sources</td><td>0.46</td><td>0.57</td><td>0.46</td><td>0.51</td><td>0.45</td><td>0.51</td><td>0.45</td><td>0.54</td><td>0.23</td><td>0.30</td><td>0.23</td><td>0.34</td></tr>
<tr><td>6</td><td>INDUS-SDE-ST</td><td>Best Stage 2: peach-night-57 + Stage 1 (anti-forgetting)</td><td>0.47</td><td>0.58</td><td>0.47</td><td>0.51</td><td><b>0.48</b></td><td><b>0.54</b></td><td><b>0.48</b></td><td>0.56</td><td>0.23</td><td>0.31</td><td>0.23</td><td>0.34</td></tr>
<tr><td>7</td><td>dutiful-thunder-110</td><td>Stage 2 (No Cite) + S1; NASA-SDE 0.75x, Tanh</td><td>0.43</td><td>0.55</td><td>0.43</td><td>0.49</td><td>0.43</td><td>0.50</td><td>0.43</td><td>0.52</td><td><b>0.26</b></td><td><b>0.34</b></td><td><b>0.26</b></td><td><b>0.37</b></td></tr>
<tr><td colspan="15"><em>Binary (1-bit) &amp; EQAT Variants</em></td></tr>
<tr><td>8</td><td>INDUS-SDE-ST-PB</td><td>1-bit post-training binarization of INDUS-SDE-ST</td><td><b>0.43</b></td><td><b>0.55</b></td><td><b>0.43</b></td><td><b>0.49</b></td><td><b>0.42</b></td><td><b>0.49</b></td><td><b>0.42</b></td><td><b>0.51</b></td><td>0.21</td><td>0.28</td><td>0.21</td><td>0.31</td></tr>
<tr><td>9</td><td>azure-eon-73</td><td>EQAT start: 1-bit, Stage 2 (No Cite), weighted</td><td>0.38</td><td>0.47</td><td>0.38</td><td>0.41</td><td>0.25</td><td>0.31</td><td>0.25</td><td>0.33</td><td>0.21</td><td>0.27</td><td>0.21</td><td>0.30</td></tr>
<tr><td>10</td><td>absurd-snowflake-85</td><td>INDUS-SDE-ST config + 1-bit EQAT</td><td>0.41</td><td>0.51</td><td>0.41</td><td>0.44</td><td>0.28</td><td>0.34</td><td>0.28</td><td>0.36</td><td>0.21</td><td>0.28</td><td>0.21</td><td>0.30</td></tr>
<tr><td>11</td><td>INDUS-SDE-ST-EQAT</td><td>Best EQAT: Stage 2 (No Cite) + S1; NASA-SDE 0.75x</td><td>0.41</td><td>0.51</td><td>0.41</td><td>0.44</td><td>0.32</td><td>0.38</td><td>0.32</td><td>0.40</td><td>0.22</td><td>0.29</td><td>0.22</td><td>0.31</td></tr>
<tr><td>12</td><td>eternal-energy-94</td><td>ST-EQAT config, NASA-SDE removed</td><td>0.40</td><td>0.51</td><td>0.40</td><td>0.44</td><td>0.30</td><td>0.37</td><td>0.30</td><td>0.39</td><td>0.10</td><td>0.13</td><td>0.10</td><td>0.14</td></tr>
<tr><td>13</td><td>wandering-snowflake-98</td><td>ST-EQAT config, weight_decay 0.1</td><td>0.41</td><td>0.51</td><td>0.41</td><td>0.45</td><td>0.29</td><td>0.36</td><td>0.29</td><td>0.38</td><td>0.21</td><td>0.28</td><td>0.21</td><td>0.31</td></tr>
<tr><td>14</td><td>dutiful-thunder-PB</td><td>1-bit post-binarization of thunder-110 (Tanh)</td><td>0.41</td><td>0.51</td><td>0.41</td><td>0.45</td><td>0.37</td><td>0.42</td><td>0.37</td><td>0.44</td><td><b>0.23</b></td><td><b>0.31</b></td><td><b>0.23</b></td><td><b>0.34</b></td></tr>
</tbody>
</table>

*INDUS-SDE-ST is the best Stage-2 FP32 model; INDUS-SDE-ST-EQAT is the deployed 1-bit variant (10–16× storage reduction).*

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

## Citation

If you use INDUS-SDE, INDUS-SDE-ST, or these workflows in your research, please cite:

```bibtex
@inproceedings{pantha2026indussde,
  author    = {Pantha, Nishan and Awale, Sajil and Kuruvanthodi, Vishnudev and KC, Simran and Ramasubramanian, Muthukumaran and Davis, Carson and Praveen, Bishwas and Foshee, Emily and Bhattacharjee, Bishwaranjan and Bugbee, Kaylin and Ramachandran, Rahul},
  title     = {{INDUS-SDE}: A Language Model for Scientific Content Curation and Discovery},
  year      = {2026},
  isbn      = {979-8-4007-2259-2},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3770855.3818847},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '26)},
  location  = {Jeju Island, Republic of Korea},
  series    = {KDD '26}
}
```
