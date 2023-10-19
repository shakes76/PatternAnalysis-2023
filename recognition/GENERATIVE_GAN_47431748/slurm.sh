#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=COMP3710_ADNI_BRAIN
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=All
#SBATCH --mail-user=j.cashman@uqconnect.edu.au
#SBATCH -o VQVAE_out.txt
#SBATCH -e VQVAE_err.txt
#SBATCH --partition=vgpu
#SBATCH --gres=gpu:1

conda activate tf-2

python ~/COMP3710/train.py
python ~/COMP3710/predict.py