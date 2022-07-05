#!/bin/bash
#SBATCH --job-name="mpn-mots-eval"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx_8000:1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=60:00:00
#SBATCH --output=output/slurm/logs/slurm-%j.txt
#SBATCH --error=output/slurm/logs/slurm-%j.txt
srun python scripts/evaluate.py
