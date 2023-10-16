#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=yolov7_ISIC2017
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=All
#SBATCH --mail-user=l.moan@uqconnect.edu.au
#SBATCH -o out.txt
#SBATCH -e err.txt
#SBATCH --partition=test
#SBATCH --gres=gpu:1

# Activate Python environment (optional)
conda activate torch-gpu

python3 yolov7/train.py  --workers 8  --device 0 --batch-size 2 --data data/ISIC_2017_downsampled/isic.yaml --img 32 32 --cfg yolov7_isic_cfg.yaml --weights '' --name yolov7 --hyp hyp.scratch.p5.yaml