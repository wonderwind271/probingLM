#!/bin/bash

#SBATCH --job-name=native_lens
#SBATCH --time=12:00:00
#SBATCH --mem=160GB
#SBATCH --partition=COMPLING
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2

#SBATCH -o JOB-%j.out
#SBATCH -e JOB-%j.err

#SBATCH --export=ALL

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "training native lens with multiGPU"

# /u501/x25luo/.conda/envs/grounding/bin/python -c "import time; time.sleep(3600*12)"
/u501/x25luo/.conda/envs/grounding/bin/python src/train_native_lens.py
