#!/bin/bash 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=JunHyukKimReport
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=All
#SBATCH --mail-user=my_email_address
#SBATCH -o JunHyukKimReportOutput.txt
#SBATCH -e JunHyukKimReportError.txt
#SBATCH --partition=vgpu
#SBATCH --gres=gpu:1

python3 ~/predict.py