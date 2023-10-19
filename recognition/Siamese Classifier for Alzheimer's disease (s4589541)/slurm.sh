#!/bin/bash
time=0-00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --job-name="siamese"
#SBATCH --mail-user=s4589541@uq.edu.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH —output=output_dir/%j.out
module load pytorch/2.0.1
conda activate /home/Student/s4589541/miniconda3/envs/venv
python /home/Student/s4589541/comp3710/report/PatternAnalysis-2023/recognition/Siamese Classifier for Alzheimer's disease (s4589541)/train.py