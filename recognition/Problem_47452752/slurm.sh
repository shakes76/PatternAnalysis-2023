#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --gres=gpu:1 
#SBATCH --partition=test 
#SBATCH --job-name="COMP3710 Report"
#SBATCH --mail-user=s4745275@student.uq.edu.au
#SBATCH --mail-type=ALL 
#SBATCH -e test_err.txt
#SBATCH -o test_out.txt

source /home/Student/s4745275/miniconda/bin/activate /home/Student/s4745275/my_demo_environment
python train.py
