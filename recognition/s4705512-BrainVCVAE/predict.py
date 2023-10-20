"""
predict.py
Author: Francesca Brzoskowski
s4705512
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

ds = Data()

# Create the model and load the weights
train_ds = ds.get_train_dataset()
data_variance = np.var(train_ds)
test_ds = ds.get_test_dataset()

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

# Encode the given images
def encode_images(images):
    """
    Encodes Images
    Parameters
    ----------
    Image : tf.Tensor
        Image from Dataset

    Returns
    -------
    encoded outputs
        """
    encoded_outputs = encoder.predict(images)
    return encoded_outputs

# Decode the given codes
def decode_images(codes):
     """
    Decodes coded Images back into images
    Parameters
    ----------
    Coded Images
        

    Returns
    -------
    Image
    """
     decoded_images = decoder.predict(codes)
     return decoded_images

# Encode and then decode images
def reconstruct(images):
    """
    Reconstructs image through trained model
    Parameters
    ----------
    Image : 
        Image to be reconstructed

    Returns
    -------
    new images reconstructed
        """
    return trained_vqvae_model.predict(images)

# Generate codes to be decoded into novel brains
def generate_codes(num_codes=4):
    """
    Generated new images using the Pixel CNN
    ----------
    num_codes : int
        number of new images to generate

    Returns
    -------
    new images
    """
    # Generate new images with the PixelCNN model
    inputs = layers.Input(shape=pixel_cnn.input_shape[1:])
    outputs = pixel_cnn(inputs, training=False)
    categorical_layer = tfp.layers.DistributionLambda(tfp.distributions.Categorical)
    outputs = categorical_layer(outputs)
    sampler = tf.keras.Model(inputs, outputs)

    # Create an empty array of priors.
    priors = np.zeros(shape=(num_codes,) + (pixel_cnn.input_shape)[1:])
    num_codes, rows, cols = priors.shape

    # Iterate over the priors because generation has to be done sequentially pixel by pixel.
    for row in range(rows):
        for col in range(cols):
            # Feed the whole array and retrieving the pixel value probabilities for the next
            # pixel.
            probs = sampler.predict(priors)
            # Use the probabilities to pick pixel values and append the values to the priors.
            priors[:, row, col] = probs[:, row, col]

    print(f"Prior shape: {priors.shape}")

    # Perform an embedding lookup.
    pretrained_embeddings = quantizer.embeddings
    priors_ohe = tf.one_hot(priors.astype("int32"), vqvae_trainer.num_embeddings).numpy()
    quantized = tf.matmul(
        priors_ohe.astype("float32"), pretrained_embeddings, transpose_b=True
    )
    quantized = tf.reshape(quantized, (-1, *(encoder.output_shape[1:])))

    return priors, quantized

def generate(num):
    """
    Generated new images using generating codes and decoding
    Plots and shows the new images on plots
    ----------
    num : int
        number of new images to generate

    """
    if num == None or 0:
        num_generations = 4
    else:
        num_generations = num
    # Generate novel images.
    priors, codes = generate_codes(num_generations)
    generated_samples = decode_images(codes)

    plt.figure(figsize=(8, num_generations * 4))
    plt.subplot(num_generations, 2, 1)
    plt.title("Code")
    plt.subplot(num_generations, 2, 2)
    plt.title("Generated Sample")

    for i in range(num_generations):
        plt.subplot(num_generations, 2, i * 2 + 1)
        plt.imshow(priors[i], cmap='gray')
        plt.axis("off")

        plt.subplot(num_generations, 2, i * 2 + 2)
        plt.imshow(generated_samples[i].squeeze() + 0.5, cmap='gray')
        plt.axis("off")
    plt.tight_layout()
    plt.savefig('generated.png')
    plt.close()

def reconstruct():
    """
    Reconstructs  images from generating codes
    Plots and shows the  images on plots
    ----------

    """
    num_images = 4

    trained_vqvae_model = vqvae_trainer.vqvae
    idx = np.random.choice(len(test_ds), num_images)
    images = test_ds[idx]
    encoded = encode_images(images)
    decoded = decode_images(encoded)


    # Quantize the encoded images
    flat_enc_outputs = encoded.reshape(-1, encoded.shape[-1])
    codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
    codebook_indices = codebook_indices.numpy().reshape(encoded.shape[:-1])

    # Display the original, encoded, and decoded images
    plt.figure(figsize=(12, num_images * 4))
    plt.subplot(num_images, 3, 1)
    plt.title("Original")
    plt.subplot(num_images, 3, 2)
    plt.title("Code")
    plt.subplot(num_images, 3, 3)
    plt.title("Decoded")

    for i in range(num_images):
        plt.subplot(num_images, 3, i * 3 + 1)
        plt.imshow(images[i].squeeze() + 0.5, cmap='gray')
        plt.axis("off")

        plt.subplot(num_images, 3, i * 3 + 2)
        plt.imshow(codebook_indices[i] + 0.5, cmap='gray')
        plt.axis("off")

        plt.subplot(num_images, 3, i * 3 + 3)
        plt.imshow(decoded[i].squeeze() + 0.5, cmap='gray')
        plt.axis("off")
        plt.tight_layout()
        plt.savefig('original_encoded_and_decoded_images.png')
    plt.close()

if __name__ == "__main__":
    generate()
    # reconstruct from encoded images
    reconstruct()