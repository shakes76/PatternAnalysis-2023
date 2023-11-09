#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=ISIC_test
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=All
#SBATCH --mail-user=l.moan@uqconnect.edu.au
#SBATCH -o out_test.txt
#SBATCH -e err_test.txt
#SBATCH --partition=p100
#SBATCH --gres=gpu:1

# Activate Python environment (optional)
conda activate torch-gpu
module load cuda

#Checks if the weights are present in the directory
if [ ! -f "best.pt" ]; then
    echo "File does not exist, downloading..."
    python3 - <<END
import gdown

url = "https://drive.google.com/uc?id=1NiVwOQVGDGuy39O3KtKdGOj4JEyHlbV5"

output = "best.pt"
gdown.download(url, output, quiet=False)
END
else
    echo "File already exists."
fi


python yolov7/test.py --data data/ISIC_2017_0.5/isic.yaml --img 512 --batch 9 --conf 0.001 --iou 0.8 --device 0 --weights 'best.pt' --name yolov7_b32_testing