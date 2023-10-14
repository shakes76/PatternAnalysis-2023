##################################
#
# Author: Joshua Wallace
# SID: 45809978
#
###################################

import os

ENV='local'

# Dataset parameters
ADNI_ROOT_DIR = './AD_NC'


# Running Parameters
VQVAE_RETRAIN = False
GAN_RETRAIN = False


# VQVAE Model Parameters
VQVAE_LR = 1e-3
VQVAE_WD = 1e-5
VQVAE_EPOCHS = 10
BATCH_SIZE = 32
VQVAE_SAVEPATH = os.path.join(os.getcwd(), 'models/vqvae')
VQVAE_MODEL_PATH = os.path.join(os.getcwd(), 'models/vqvae/vqvae.pth')

GAN_LR = 1e-3
GAN_WD = 1e-5
GAN_EPOCHS = 10
GAN_SAVEPATH = os.path.join(os.getcwd(), 'models/gan')
DISCRIMINATOR_PATH = os.path.join(os.getcwd(), 'models/gan/gan_discriminator.pth')
GENERATOR_PATH = os.path.join(os.getcwd(), 'models/gan/gan_generator.pth')

# Prediction Parameters
OUTPUT_PATH = os.path.join(os.getcwd(), 'models/predictions/output')
NUM_IMAGES = 1