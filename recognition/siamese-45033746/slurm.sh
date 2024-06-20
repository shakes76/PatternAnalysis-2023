#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:1
#SBATCH --partition=vgpu
#SBATCH --job-name="uwu-train"
#SBATCH --account=s4503374
#SBATCH --mail-user=treffery.webb@uqconnect.edu.au
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
conda activate /home/Student/s4503374/patterns/env
python /home/Student/s4503374/PatternAnalysis-2023/recognition/siamese-45033746/train.py