#!/bin/bash
#SBATCH --partition=p100
#SBATCH --gres=gpu:1
#SBATCH --job-name=ADNIVIT
#SBATCH --output=ADNIVIT.out

source /home/Student/s4581696/.bashrc
conda activate /home/Student/s4581696/coco/envs/torch2
export PYTHONPATH="/home/Student/s4581696/slurm"
echo $PYTHONPATH
python3 ./vit_rangpur.py
