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
from dataset import *
from modules import *

image_size = (128, 128)
latent_dim = 16
num_embeddings = 128
vqvae_epochs = 30
vqvae_batch_size = 128

# PixelCNN hyperparameters
num_residual_blocks = 2
num_pixelcnn_layers = 2
pixelcnn_epochs = 30
pixelcnn_batch_size = 128
pixelcnn_validation_split = 0.1
models_directory = "trained_models/"
vqvae_weights_filename = "vqvae/vqvae"
pixelcnn_weights_filename = "pixelcnn/pixelcnn"

# Make sure the trained weights exist
if not os.path.isfile(models_directory + vqvae_weights_filename + ".index"):
    print("ERROR: Missing VQ-VAE training weights. Please run train.py", file=sys.stderr)
    exit(1)
if not os.path.isfile(models_directory + pixelcnn_weights_filename + ".index"):
    print("ERROR: Missing PixelCNN training weights. Please run train.py", file=sys.stderr)
    exit(1)

# Load testing dataset
ds = Data
test_ds = ds.get_test_dataset()

# Create the model and load the weights
train_ds = ds.get_train_dataset()
data_variance = np.var(train_ds)
vqvae_trainer = VQVAETrainer(
        data_variance,
        latent_dim=latent_dim,
        num_embeddings=num_embeddings,
)


vqvae_trainer.load_weights(models_directory + vqvae_weights_filename)

encoder = vqvae_trainer.vqvae.get_layer("encoder")
quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")
decoder = vqvae_trainer.vqvae.get_layer("decoder")

# Load the PixelCNN model
pixelcnn_input_shape = quantizer.output_shape[1:-1]
print(f"Input shape of the PixelCNN: {pixelcnn_input_shape}")

pixel_cnn = get_pixelcnn(
        num_residual_blocks,
        num_pixelcnn_layers,
        pixelcnn_input_shape,
        vqvae_trainer.num_embeddings,
)
pixel_cnn.load_weights(models_directory + pixelcnn_weights_filename)
