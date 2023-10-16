"""
predict.py
Author: Francesca Brzoskowski

Shows example usage of the trained model. 
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp

from dataset import get_test_dataset, get_train_dataset, scale_image, image_size
from modules import VQVAETrainer, get_pixelcnn
from hyperparameters import *
from utils import models_directory, vqvae_weights_filename, pixelcnn_weights_filename

# Make sure the trained weights exist
if not os.path.isfile(models_directory + vqvae_weights_filename + ".index"):
    print("ERROR: Missing VQ-VAE training weights. Please run train.py", file=sys.stderr)
    exit(1)
if not os.path.isfile(models_directory + pixelcnn_weights_filename + ".index"):
    print("ERROR: Missing PixelCNN training weights. Please run train.py", file=sys.stderr)
    exit(1)

# Load testing dataset
test_ds = get_test_dataset()

# Create the model and load the weights
train_ds = get_train_dataset()
data_variance = np.var(train_ds)
vqvae_trainer = VQVAETrainer(
        data_variance,
        latent_dim=latent_dim,
        num_embeddings=num_embeddings,
)