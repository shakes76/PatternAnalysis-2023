#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=YOLOV7_p6
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=All
#SBATCH --mail-user=l.moan@uqconnect.edu.au


#SBATCH --partition=p100
#SBATCH --gres=gpu:1

# Activate Python environment (optional)
conda activate torch-gpu
module load cuda

python3 yolov7/train.py  --device 0 --workers 8  --batch-size 128 --data data/ISIC_2017_0.5/isic.yaml --img 512 512 --cfg yolov7_isic_cfg.yaml --weights 'yolov7_training.pt' --name YOLOVV7_P100_p5 --hyp hyp.scratch.p5.yaml