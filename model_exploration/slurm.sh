#!/bin/bash
#
#SBATCH --mail-user=sa0812@uah.edu 
#SBATCH --job-name=llm_slurm_conda       # Job name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --gres=gpu:a100:2                # Request 2 GPUs (A100)
#SBATCH --cpus-per-task=16               # Number of CPU cores per task
#SBATCH --mem=256G                        # Total memory
#SBATCH --output=slurm_logs/%j_%x.out           # Standard output
#SBATCH --error=slurm_logs/%j_%x.err            # Standard error
#SBATCH --time=14-0:00:00                 # Walltime hh:mm:ss
#SBATCH --ntasks-per-node=1              # Number of tasks per node
#SBATCH --mail-type=END,FAIL


export GPUS_PER_NODE=2
export OMP_NUM_THREADS=1

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
      

echo "===== SLURM ENVIRONMENT ====="
scontrol show hostnames $SLURM_JOB_NODELIST
echo "Using $SLURM_NNODES nodes"
echo "MASTER_ADDR:PORT = $MASTER_ADDR:$MASTER_PORT"
echo "GPUS_PER_NODE = $GPUS_PER_NODE"

echo "Job ID:    $SLURM_JOB_ID"
echo "Node List: $SLURM_NODELIST"
echo "CPUs:      $SLURM_CPUS_ON_NODE"
echo

echo "===== BEFORE LOADING CONDA ====="
echo "PATH:      $PATH"
echo "Which python: $(which python)"
echo

echo "===== LOAD CONDA VIA SHELL HOOK ====="
# Bootstraps conda in a non-interactive shell
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate slurm-test

echo
echo "===== AFTER ACTIVATION ====="
echo "PATH:      $PATH"
echo "Which python: $(which python)"
python --version
echo

echo "===== PYTHON SMOKE TEST ====="
python - << 'PYCODE'
import sys
print("Hello from Conda Python!")
print("sys.executable:", sys.executable)
print("sys.path sample:", sys.path[:3])
PYCODE

echo
echo "===== TORCH GPU TEST ====="
python - << 'PYCODE'
import torch, os

# Basic CUDA / env info
print("CUDA available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "all"))

if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f"Number of GPUs detected by torch: {n}")
    for idx in range(n):
        name = torch.cuda.get_device_name(idx)
        print(f"  GPU {idx}: {name}")
    current = torch.cuda.current_device()
    print(f"Current torch device index: {current}")
    print(f"Current torch device name: {torch.cuda.get_device_name(current)}")
else:
    print("No GPUs accessible to torch.")
PYCODE

echo
echo ">>>>>> test python file"
# torchrun --nproc-per-node=2 st_trainer_ddp_hf.py

srun --mem=0 torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_endpoint="$MASTER_ADDR":"$MASTER_PORT" \
    --rdzv_backend=c10d \
    st_trainer_ddp_hf.py --batch_size 32 --gradient_accumulation_steps 16 --lr 3.0e-5
echo "<<<<<< test python file"
