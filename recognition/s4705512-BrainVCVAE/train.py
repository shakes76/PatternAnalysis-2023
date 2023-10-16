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



def save_subplot(original, reconstructed,i):
    plt.subplot(1, 2, 1)
    plt.imshow(original.squeeze() + 0.5)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.squeeze() + 0.5)
    plt.title("Reconstructed")
    plt.axis("off")

    plt.savefig((str(i) + '.png'))
    plt.close()


trained_vqvae_model = vqvae_trainer.vqvae
idx = np.random.choice(len(test), 10)
test_images = test[idx]
reconstructions_test = trained_vqvae_model.predict(test_images)

i=0
for test_image, reconstructed_image in zip(test_images, reconstructions_test):
    save_subplot(test_image, reconstructed_image,i)
    i+=1

# Save the model
vqvae_trainer.save_weights(models_directory + vqvae_weights_filename)

# load in the model
# Assuming vqvae_trainer is your VQVAETrainer instance


# Load the saved weights
vqvae_trainer.load_weights(models_directory + vqvae_weights_filename)
vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
vqvae_history = vqvae_trainer.fit(train, epochs=30, batch_size=128)

# Set up the PixelCNN to generate images that imitate the code, to generate
# new brains
encoder = vqvae_trainer.vqvae.get_layer("encoder")
quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")
pixelcnn_input_shape = quantizer.output_shape[1:-1]
print(f"Input shape of the PixelCNN: {pixelcnn_input_shape}")


pixel_cnn = get_pixelcnn(
        num_residual_blocks,
        num_pixelcnn_layers,
        pixelcnn_input_shape,
        vqvae_trainer.num_embeddings,
)

# Generate the codebook indices. Only do it on half the training set to avoid memory issues
encoded_outputs = encoder.predict(train[:len(train) // 2], batch_size=128)
flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
codebook_indices = quantizer.get_code_indices(flat_enc_outputs)

codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
print(f"Shape of the training data for PixelCNN: {codebook_indices.shape}")

