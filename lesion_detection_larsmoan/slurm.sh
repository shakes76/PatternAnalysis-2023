#!/bin/bash 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=torch-gpu
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=All
#SBATCH --mail-user=l.moan@uqconnect.edu.au
#SBATCH -o out.txt
#SBATCH -e err.txt
#SBATCH --partition=test
#SBATCH --gres=gpu:1

conda activate gpu_fix

python check_torch.py