#!/bin/bash

#SBATCH --job-name=test_native_len
#SBATCH --time=12:00:00
#SBATCH --mem=100GB
#SBATCH --partition=ALL
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

#SBATCH -o JOB-%j.out
#SBATCH -e JOB-%j.err

# /u501/x25luo/.conda/envs/grounding/bin/python -c "import time; time.sleep(3600*8)"
/u501/x25luo/.conda/envs/grounding/bin/python src/inference_native_len.py