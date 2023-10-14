##################################
#
# Author: Joshua Wallace
# SID: 45809978
#
###################################

import os
import torch

ENV='local'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset parameters
if ENV == 'local':
    ADNI_ROOT_DIR = os.path.join(os.getcwd(), 'AD_NC')
else :
    ADNI_ROOT_DIR = '/home/groups/comp3710/ADNI'

CHANNELS = 3
W = 128
H = 128
BETA = 0.25
FRACTION = 1.0



# Running Parameters
VQVAE_RETRAIN = True
GAN_RETRAIN = False
VQVAE_TEST = True
VQVAE_PREDICT = True


# VQVAE Model Parameters
VQVAE_HIDDEN = 128
VQVAE_RESIDUAL = VQVAE_HIDDEN // 4
VQVAE_EMBEDDINGS = VQVAE_HIDDEN * 4
VQVAE_EMBEDDING_DIM = VQVAE_HIDDEN // 2

VQVAE_LR = 1e-3
VQVAE_WD = 1e-5
VQVAE_EPOCHS = 100
BATCH_SIZE = 32

VQVAE_SAVEPATH = os.path.join(os.getcwd(), 'models/vqvae')
VQVAE_MODEL_PATH = os.path.join(os.getcwd(), 'models/vqvae/vqvae.pth')
VQVAE_RECONSTRUCT_PATH = os.path.join(os.getcwd(), 'models/predictions/vqvae_reconstruction.png')


GAN_LR = 1e-3
GAN_WD = 1e-5
GAN_EPOCHS = 10
GAN_SAVEPATH = os.path.join(os.getcwd(), 'models/gan')
DISCRIMINATOR_PATH = os.path.join(os.getcwd(), 'models/gan/gan_discriminator.pth')
GENERATOR_PATH = os.path.join(os.getcwd(), 'models/gan/gan_generator.pth')

# Prediction Parameters
OUTPUT_PATH = os.path.join(os.getcwd(), 'models/predictions/output_new_')
NUM_IMAGES = 2