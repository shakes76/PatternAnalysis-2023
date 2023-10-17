#!/bin/bash
#SBATCH --partition=vgpu20
#SBATCH --gres=gpu:1
#SBATCH --job-name=ADNIVIT
#SBATCH --output=vit11.out

source /home/Student/s4581696/.bashrc
conda activate /home/Student/s4581696/coco/envs/torch2
export PYTHONPATH="/home/Student/s4581696/slurm"
echo $PYTHONPATH
python3 ./vit_rangpur2.py
