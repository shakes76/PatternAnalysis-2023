#!/bin/bash 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=Stylegan
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=All
#SBATCH --mail-user=a.fittler@uqconnect.edu.au
#SBATCH -o stylegan_out.txt
#SBATCH -e stylegan_err.txt
#SBATCH --partition=vgpu
#SBATCH --gres=gpu:1

conda activate pytorch

python predict.py
