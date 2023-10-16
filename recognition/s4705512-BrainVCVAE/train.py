"""
train.py
Author: Francesca Brzoskowski

Contains the source code for training, validating, testing and saving the model. The model
should is imported from “modules.py” and the data loader is imported from “dataset.py”. 
Losses are plotted from training
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from modules import VQVAETrainer, get_pixelcnn
from dataset import Data 
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

# Get the Data
# Get all attributes of the class
ds = Data()
train = ds.get_train_dataset()
test = ds.get_test_dataset()
data_variance = np.var(train / 255.0)

# print("train", "size", train.size, "shape", train.shape)
# print("test", "size", test.size, "shape", test.shape)


# training implmented from https://keras.io/examples/generative/vq_vae/
vqvae_trainer = VQVAETrainer(data_variance, latent_dim=16, num_embeddings=128)
vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
vqvae_history = vqvae_trainer.fit(train, epochs=30, batch_size=128)
