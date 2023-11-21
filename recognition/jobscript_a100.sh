#!/bin/bash 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name="s4663144-siamese-a100"
#SBATCH --cpus-per-task 1
#SBATCH -o s4663144-siamese-a100_out.txt
#SBATCH -e s4663144-siamese-a100_err.txt
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mail-user=j.lukas@uqconnect.edu.au   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
##conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
##conda activate conda-torch
conda install tensorflow keras
python train.py