#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=ISIC_test
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=All
#SBATCH --mail-user=l.moan@uqconnect.edu.au
#SBATCH -o out_test.txt
#SBATCH -e err_test.txt
#SBATCH --partition=test
#SBATCH --gres=gpu:1

# Activate Python environment (optional)
conda activate torch-gpu
module load cuda


# Define the filename and file URL
filename="best.pt"
file_url="https://drive.google.com/uc?id=1NiVwOQVGDGuy39O3KtKdGOj4JEyHlbV5"

# Check if the file exists in the current directory
if [ -e "$PWD/$filename" ]; then
    echo "File '$filename' already exists in the current directory."
else
    # If file doesn't exist, download it
    echo "Downloading '$filename'..."
    wget "$file_url"
    echo "Download complete."
fi


python yolov7/test.py --data data/ISIC_2017_0.5/isic.yaml --img 512 --batch 2 --conf 0.001 --iou 0.65 --device 0 --weights 'best.pt' --name yolov7_b32_testing
