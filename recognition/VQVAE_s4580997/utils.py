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

# Dataset location
if ENV == 'local':
    ADNI_ROOT_DIR = os.path.join(os.getcwd(), 'AD_NC')
else :
    ADNI_ROOT_DIR = '/home/groups/comp3710/ADNI/AD_NC'

# Image.Dataset parameters
CHANNELS = 3
W = 256
H = 256
BETA = 0.25
FRACTION = 0.1

# Running Parameters
VQVAE_RETRAIN = True
GAN_RETRAIN = True
PIXELCNN_RETRAIN = False
VQVAE_TEST = True
VQVAE_PREDICT = True

# VQVAE Model Parameters
VQVAE_HIDDEN = 128
VQVAE_RESIDUAL = VQVAE_HIDDEN // 4
VQVAE_EMBEDDINGS = VQVAE_HIDDEN * 4
VQVAE_EMBEDDING_DIM = VQVAE_HIDDEN // 2

# VQVAE Hyperparameters
VQVAE_LR = 1e-3
VQVAE_WD = 1e-5
VQVAE_EPOCHS = 2
BATCH_SIZE = 32

# VQVAE Paths
VQVAE_SAVEPATH = os.path.join(os.getcwd(), 'models/vqvae')
VQVAE_MODEL_PATH = os.path.join(os.getcwd(), 'models/vqvae/vqvae.pth')
VQVAE_RANGPUR_MODEL_PATH = os.path.join(os.getcwd(), 'models/rangpur/new/vqvae.pth')
VQVAE_RECONSTRUCT_PATH = os.path.join(os.getcwd(), 'models/predictions/vqvae_reconstruction.png')

# GAN Model Parameters
GAN_LATENT_DIM = 128
GAN_IMG_SIZE = 64
NOISE = 100

# GAN Hyperparameters
GAN_LR = 1e-3
GAN_WD = 1e-5
GAN_EPOCHS = 20
GAN_SAVEPATH = os.path.join(os.getcwd(), 'models/gan')
DISCRIMINATOR_MODEL_PATH = os.path.join(os.getcwd(), 'models/gan/gan_discriminator.pth')
DISCRIMINATOR_RANGPUR_MODEL_PATH = os.path.join(os.getcwd(), 'models/rangpur/new/discriminator.pth')
GENERATOR_MODEL_PATH = os.path.join(os.getcwd(), 'models/gan/gan_generator.pth')
GENERATOR_RANGPUR_MODEL_PATH = os.path.join(os.getcwd(), 'models/rangpur/new/generator.pth')

# PixelCNN
PIXEL_MODEL_PATH=os.path.join(os.getcwd(), 'models/pixelcnn.pth')
PIXEL_RANGPUR_MODEL_PATH = os.path.join(os.getcwd(), 'models/rangpur/new/pixelcnn.pth')
PIXEL_SAVEPATH=os.path.join(os.getcwd(), 'models/pixelcnn')



# Prediction Parameters
OUTPUT_PATH = os.path.join(os.getcwd(), 'models/predictions/generated_')
NUM_IMAGES = 32

# Misc
PRINT_AT = 10
