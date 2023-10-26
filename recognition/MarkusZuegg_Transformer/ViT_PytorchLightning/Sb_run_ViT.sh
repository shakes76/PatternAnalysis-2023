#!/bin/bash

#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:1
#SBATCH --partition=vgpu
#SBATCH --job-name="ViT"
#SBATCH --mail-user=markusz2020@gmail.com
# SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=x%j.out
#SBATCH --error=x%j.err

module load cuda
conda activate pytorch2

srun python train.py
