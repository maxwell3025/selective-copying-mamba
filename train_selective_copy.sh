#!/bin/bash
#SBATCH -c 6  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p gpu-preempt  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 01:30:00  # Job time limit
#SBATCH -o train_mamba_%j.out  # %j = job ID
#SBATCH --constraint sm_70  # %j = job ID

mkdir -p /work/pi_jaimedavila_umass_edu/maxwelltang_umass_edu/selective-copying-mamba-maxwell3025
cd /work/pi_jaimedavila_umass_edu/maxwelltang_umass_edu/selective-copying-mamba-maxwell3025
module load cuda/12.2.1
if ! test -d .venv; then
    module load python/3.11.0
    python3 -m venv .venv
    module unload python/3.11.0
fi

source ./.venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install wheel packaging setuptools
python -m pip install causal-conv1d
python -m pip install mamba-ssm
python -m pip install wandb
export WANDB_API_KEY=6701a2ec1a7ccf30a6e05cffd0a9281b1de3b41c
python selective_copying_mamba.py
# wget https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu122torch2.4cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
# wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu122torch2.4cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
# pip install mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# pip install mamba_ssm-2.2.2+cu122torch2.4cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
# pip install causal_conv1d-1.4.0+cu122torch2.4cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

# python -m pip install
# export WANDB_API_KEY=6701a2ec1a7ccf30a6e05cffd0a9281b1de3b41c

# python -m train pipeline=mnist model=s4
