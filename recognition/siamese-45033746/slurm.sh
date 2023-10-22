#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --job-name="uwu"
#SBATCH --mail-user=treffery.webb@uqconnect.edu.au
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH —output=output_dir/%j.out
module load pytorch/2.0.1
conda activate /home/Student/s4503374/patterns/env
python /home/Student/s4503374/PatternAnalysis-2023/recognition/siamese-45033746/train.py