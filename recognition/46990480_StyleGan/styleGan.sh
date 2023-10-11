#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=vgpu
#SBATCH --job-name="COMP3710"
#SBATCH --mail-user=s4699048@student.uq.edu.au
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH -o /home/Student/s4699048/out_styleGan/output_%j.out
#SBATCH -e /home/Student/s4699048/out_styleGan/error_%j.out

echo "Selecting Environment"
conda activate comp3710
echo "Environment Selected"

echo "Starting Script"
python ~/dataset.py
echo "Script Finished"
