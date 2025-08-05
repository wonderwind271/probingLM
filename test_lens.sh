#!/bin/bash

#SBATCH --job-name=grounding
#SBATCH --time=12:00:00
#SBATCH --mem=10GB
#SBATCH --partition=ALL
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

#SBATCH -o JOB-%j.out
#SBATCH -e JOB-%j.err

/u501/x25luo/.conda/envs/grounding/bin/python src/inference_vocab_len.py