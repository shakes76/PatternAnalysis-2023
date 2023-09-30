#!/bin/bash
#SBATCH --job-name=Report
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ppalanivelurajmohan@gmail.com
#SBATCH --partition=vgpu
#SBATCH --gres=gpu:1

conda activate new_base
python modules.py
