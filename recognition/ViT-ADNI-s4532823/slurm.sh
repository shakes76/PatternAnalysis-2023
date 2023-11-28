#!/bin/bash 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=VitADNI
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=All
#SBATCH --mail-user=l.blaine@uq.net.au
#SBATCH -o vitadni_out.txt
#SBATCH -e vitadni_err.txt
#SBATCH --partition=p100
#SBATCH --gres=gpu:1

conda activate conda-env

cd ~/COMP3710/report/PatternAnalysis-2023/recognition/ViT-ADNI-s4532823
python train.py
