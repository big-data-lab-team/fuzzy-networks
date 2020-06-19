#!/bin/bash
#
#SBATCH --job-name=fuzzy-dl-training
#SBATCH --output=sbatch-output.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

srun python3 train.py mlp
