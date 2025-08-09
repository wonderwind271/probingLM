#!/bin/bash

#SBATCH --job-name=native_lens
#SBATCH --time=12:00:00
#SBATCH --mem=160GB
#SBATCH --partition=COMPLING
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2

#SBATCH -o JOB-%j.out
#SBATCH -e JOB-%j.err

#SBATCH --export=ALL

export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=1

echo "training native lens with multiGPU"
/u501/x25luo/.conda/envs/grounding/bin/python src/train_native_lens.py