#!/bin/bash
#
#SBATCH --job-name=fuzzy-dl-training
#SBATCH --output=sbatch-output-cifar10.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

srun python3 benchmark.py
