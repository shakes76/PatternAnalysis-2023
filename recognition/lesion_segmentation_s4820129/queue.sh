#!/bin/bash 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=UNET
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=All
#SBATCH --mail-user=i.wangensteen@uq.net.au
#SBATCH -o out.txt
#SBATCH -e err.txt
#SBATCH --partition=test
#SBATCH --gres=gpu:1

conda activate cluster_env

python3 ~/PatternAnalysis-2023/lesion_segmentation_s4820129/train.py