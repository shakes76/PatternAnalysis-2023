#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=p6_p100_b64
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=All
#SBATCH --mail-user=l.moan@uqconnect.edu.au
#SBATCH --partition=test
#SBATCH --gres=gpu:1

# Activate Python environment (optional)
conda activate torch-gpu
module load cuda

python3 yolov7/train.py  --device 0 --workers 8  --batch-size -1 --data data/ISIC_2017_0.5/isic.yaml --img 512 512 --cfg yolov7_isic_cfg.yaml --weights 'yolov7_training.pt' --name p6_p100_b64 --hyp hyp.scratch.p6.yaml