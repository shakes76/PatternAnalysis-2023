#!/bin/bash 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=test
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=All
#SBATCH --mail-user=s.sveen@uqconnect.edu.au
#SBATCH -o test_ass3.txt
#SBATCH -e test_err_ass3.txt
#SBATCH --partition=vgpu
#SBATCH --gres=gpu:1

conda activate conda-torch

python ~/train.py