#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=ISIC_test
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=All
#SBATCH --mail-user=l.moan@uqconnect.edu.au
#SBATCH -o out_test.txt
#SBATCH -e err_test.txt
#SBATCH --partition=test
#SBATCH --gres=gpu:1

# Activate Python environment (optional)
conda activate torch-gpu
module load cuda

python yolov7/test.py --data data/ISIC_2017_0.5/isic.yaml --img 512 --batch 2 --conf 0.001 --iou 0.65 --device 0 --weights 'best.pt' --name yolov7_b32_testing
