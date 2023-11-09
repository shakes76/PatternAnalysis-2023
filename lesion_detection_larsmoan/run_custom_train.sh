#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=p6_p100_b64
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=All
#SBATCH --mail-user=l.moan@uqconnect.edu.au
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH -o out.txt
#SBATCH -e err.txt

# Activate Python environment (optional)
conda activate torch-gpu
module load cuda

#Checks if the transfer learning weights (YOLOV7 weights (COCO)) are present in the dir, if not downloads them
if [ ! -f "yolov7_training.pt" ]; then
    echo "File does not exist, downloading..."
    python3 - <<END
import gdown

weights_url = "https://drive.google.com/uc?id=1mAu29ZlOTn3csjnZ5fmro10kY3XxAddC"
output = "yolov7_training.pt"
gdown.download(weights_url, output, quiet=False)
END
else
    echo "File already exists."
fi

python3 yolov7/train.py  --device 0 --workers 8  --batch-size 1 --data data/ISIC_2017_0.5/isic.yaml --img 512 512 --cfg yolov7_isic_cfg.yaml --weights 'yolov7_training.pt' --name p6_p100_b64 --hyp hyp.scratch.p6.yaml