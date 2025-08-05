#!/bin/bash

#SBATCH --job-name=native_lens
#SBATCH --time=12:00:00
#SBATCH --mem=200GB
#SBATCH --partition=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2

#SBATCH -o JOB-%j.out
#SBATCH -e JOB-%j.err

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1

export CUDA_LAUNCH_BLOCKING=1

echo "testing without multiGPU"
/u501/x25luo/.conda/envs/grounding/bin/python src/train_native_lens.py