#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1

# number of CPU cores per task
#SBATCH --cpus-per-task=4

#SBATCH --partition=vgpu

#SBATCH --gres=gpu:1

#SBATCH --job-name=pred_img

#SBATCH --mail-type=ALL
#SBATCH --mail-user=tong.qiu@uqconnect.edu.au

conda activate metadamagenet
python3 predict.py -m ./model/Classifier_tf.pth -t preimage_predict -d ./AD_NC/test/AD/388206_94.jpeg
python3 predict.py -m ./model/Classifier_tf.pth -t preimage_predict -d ./AD_NC/test/AD/389298_79.jpeg
python3 predict.py -m ./model/Classifier_tf.pth -t preimage_predict -d ./AD_NC/test/AD/401530_99.jpeg
python3 predict.py -m ./model/Classifier_tf.pth -t preimage_predict -d ./AD_NC/test/NC/1182968_108.jpeg
python3 predict.py -m ./model/Classifier_tf.pth -t preimage_predict -d ./AD_NC/test/NC/1253769_99.jpeg
python3 predict.py -m ./model/Classifier_tf.pth -t preimage_predict -d ./AD_NC/test/NC/1281631_103.jpeg