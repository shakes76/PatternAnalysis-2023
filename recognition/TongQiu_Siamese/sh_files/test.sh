#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1

# number of CPU cores per task
#SBATCH --cpus-per-task=4

#SBATCH --partition=vgpu

#SBATCH --gres=gpu:1

#SBATCH --job-name=test_C

#SBATCH --mail-type=ALL
#SBATCH --mail-user=tong.qiu@uqconnect.edu.au

conda activate metadamagenet
python3 predict.py -m ./model/Classifier_c.pth -t test
python3 predict.py -m ./model/Classifier_t.pth -t test
python3 predict.py -m ./model/Classifier_cf.pth -t test
python3 predict.py -m ./model/Classifier_tf.pth -t test