#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=project
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=All
#SBATCH --mail-user=s4641971@student.uq.edu.au
#SBATCH --output=test_out.out
#SBATCH -e test_err.txt
#SBATCH --partition=vgpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-05:00:00

conda activate conda-torch

python ~/project/train.py
