#!/bin/bash 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=vae
#SBATCH --cpus-per-task 1
#SBATCH -o AE_out.txt
#SBATCH --partition=vgpu
#SBATCH --gres=gpu:1

conda activate py39-torch

python -u ~/COMP3710/PatternAnalysis-2023/recognition/YOLO_45296831/train.py