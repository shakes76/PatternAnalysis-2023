#!/bin/bash
#SBATCH --time=0-00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=vgpu
#SBATCH --job-name="s4589541-siamese-train"
#SBATCH --account=s4589541
#SBATCH --mail-user=m.gardiner@uq.net.au
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH —output=./slurm_outputs/%j.out
conda activate /home/Student/s4589541/miniconda3/envs/venv
python /home/Student/s4589541/comp3710/report/PatternAnalysis-2023/recognition/Siamese\ Classifier\ for\ Alzheimer\'s\ disease\ \(s4589541\)/train.py --train
