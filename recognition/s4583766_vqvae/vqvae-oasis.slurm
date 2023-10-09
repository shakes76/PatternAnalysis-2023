#!/bin/bash
#time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=p100
#SBATCH --job-name="oasis_vqvae"
#SBATCH --mail-user=s.bates@uqconnect.edu.au
#SBATCH --mail-type=All
#SBATCH --output=out/oasis_vqvae-%j.out

source /home/Student/s4583766/miniconda3/bin/activate conda-torch
python3 dataset.py
