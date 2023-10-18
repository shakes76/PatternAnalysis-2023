#!/bin/bash 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=script
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=All
#SBATCH --mail-user=t.gunnarsli@uq.net.au
#SBATCH -o out.txt
#SBATCH -e err.txt
#SBATCH --partition=vgpu
#SBATCH --gres=gpu:1

conda activate env1
~/miniconda3/envs/env1/bin/python ~/Patternanalysis-2023/recognition/Tord_Improved_UNet/train.py