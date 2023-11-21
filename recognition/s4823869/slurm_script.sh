#!/bin/bash
#SBATCH --job-name=Yash_train
#SBATCH --mail-type=All
#SBATCH --mail-user=yash.mittal@uqconnect.edu.au
#SBATCH --partition=vgpu
#SBATCH --gres=gpu:1

conda activate conda-env

python ~/train.py

conda deactivate