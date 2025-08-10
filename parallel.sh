#!/bin/bash
#SBATCH --job-name=native_lens_ddp
#SBATCH --time=12:00:00
#SBATCH --mem=160GB
#SBATCH --partition=COMPLING
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH -o JOB-%j.out
#SBATCH -e JOB-%j.err
#SBATCH --export=ALL

# recommended envs
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1

# number of processes per node = number of GPUs requested
NPROC_PER_NODE=2

# run with torchrun (PyTorch >=1.9). torchrun will set LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
# adjust python env path as needed
# activate conda env
# source /u501/x25luo/.conda/envs/grounding/bin/activate

# run
/u501/x25luo/.conda/envs/grounding/bin/python -m torch.distributed.run --nproc_per_node=${NPROC_PER_NODE} src/train_native_lens_ddp.py
# OR (equivalently)
# /usr/bin/python -m torchrun --nproc_per_node=${NPROC_PER_NODE} src/train_native_lens_ddp.py
