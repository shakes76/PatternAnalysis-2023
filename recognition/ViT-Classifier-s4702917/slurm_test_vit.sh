#!/bin/bash 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=test_vit
#SBATCH --cpus-per-task 1
#SBATCH -t 0-09:00              # time limit: (D-HH:MM) 
#SBATCH --mail-type=END
#SBATCH --mail-user=s4702917@student.uq.edu.au
#SBATCH -o output/vit_out.txt
#SBATCH -e output/vit_err.txt
#SBATCH --partition=vgpu20
#SBATCH --gres=gpu:1

conda activate conda-env

cd ~/COMP3710/ass3/recognition/ViT-Classifier-s4702917/

python predict.py
