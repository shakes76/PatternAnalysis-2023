##################################
#
# Author: Joshua Wallace
# SID: 45809978
#
###################################

import os

ENV='local'

# Dataset parameters
if ENV == 'local':
    ADNI_ROOT_DIR = os.path.join(os.getcwd(), 'AD_NC')
else :
    ADNI_ROOT_DIR = '/home/groups/comp3710/ADNI'

CHANNELS = 3
RESIZE_WIDTH = 256
RESIZE_HEIGHT = 256
BETA = 0.25


# Running Parameters
VQVAE_RETRAIN = True
GAN_RETRAIN = False


# VQVAE Model Parameters
VQVAE_LR = 1e-3
VQVAE_WD = 1e-5
VQVAE_EPOCHS = 20
BATCH_SIZE = 32
VQVAE_FRACTION = 0.5
VQVAE_SAVEPATH = os.path.join(os.getcwd(), 'models/vqvae')
VQVAE_MODEL_PATH = os.path.join(os.getcwd(), 'models/vqvae/vqvae.pth')

GAN_LR = 1e-3
GAN_WD = 1e-5
GAN_EPOCHS = 10
GAN_SAVEPATH = os.path.join(os.getcwd(), 'models/gan')
DISCRIMINATOR_PATH = os.path.join(os.getcwd(), 'models/gan/gan_discriminator.pth')
GENERATOR_PATH = os.path.join(os.getcwd(), 'models/gan/gan_generator.pth')

# Prediction Parameters
OUTPUT_PATH = os.path.join(os.getcwd(), 'models/predictions/output_new_')
NUM_IMAGES = 5