!/bin/bash
SBATCH --job-name=initial-test
SBATCH --partition=gpu
SBATCH --nodes=1
SBATCH --ntasks=1
SBATCH --ntasks-per-node=1
SBATCH --cpus-per-task=1

python3 train.py