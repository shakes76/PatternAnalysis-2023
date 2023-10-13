import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from keras import layers
from modules import VQVAETrainer, get_pixel_cnn

# Load the VQ-VAE and Pixel CNN models
vqvae_trainer = VQVAETrainer(0.03525, latent_dim=32, num_embeddings=128)
vqvae_trainer.load_weights('recognition/VQ-VAE-46495408/checkpoint/vqvae_ckpt')
pixel_cnn = get_pixel_cnn((64, 64), 128)
pixel_cnn.load_weights('recognition/VQ-VAE-46495408/checkpoint/pixelcnn_ckpt')


# Create a mini sampler model
#inputs = layers.Input(shape=)