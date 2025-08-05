#!/bin/bash

#SBATCH --job-name=native_lens
#SBATCH --time=00:30:00
#SBATCH --mem=2GB
#SBATCH --partition=ALL
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

#SBATCH -o JOB-%j.out
#SBATCH -e JOB-%j.err

export CUDA_LAUNCH_BLOCKING=1

/u501/x25luo/.conda/envs/grounding/bin/python src/test_parallel.py