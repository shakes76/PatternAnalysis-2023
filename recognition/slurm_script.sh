#!/bin/bash
time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --job-name=""
#SBATCH --mail-user=your@email.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH —output=output_dir/%j.out

module load tensorflow/1.9.0
export PYTHONPATH=~/.local/lib/python3.6/site-packages/
source ~/.myenv/bin/activate
python train2.py