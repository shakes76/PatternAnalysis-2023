import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from modules import VQVAETrainer, get_pixel_cnn

"""
Code reference source: https://keras.io/examples/generative/vq_vae/
"""

# Load the VQ-VAE and Pixel CNN models
vqvae_trainer = VQVAETrainer(0.03525, latent_dim=32, num_embeddings=128)
vqvae_trainer.load_weights('recognition/VQ-VAE-46495408/checkpoint/vqvae_ckpt')
pixel_cnn = get_pixel_cnn((64, 64), 128)
pixel_cnn.load_weights('recognition/VQ-VAE-46495408/checkpoint/pixelcnn_ckpt')

# Create a prior to generate images
batch = 10
priors = np.zeros(shape=(batch,) + (pixel_cnn.input_shape)[1:])
batch, rows, cols = priors.shape

# Iterate over the priors
for row in range(rows):
    for col in range(cols):
        # Feed the whole array and 
        # retrieving the pixel value probabilities for the next pixel
        logits = pixel_cnn.predict(priors)
        sampler = tfp.distributions.Categorical(logits)
        probs = sampler.sample()
        priors[:, row, col] = probs[:, row, col]
        
print(f"Prior shape: {priors.shape}")

# Perform an embedding lookup
quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")
pretrained_embeddings = quantizer.embeddings
priors_ohe = tf.one_hot(priors.astype("int32"), vqvae_trainer.num_embeddings).numpy()
quantized = tf.matmul(
    priors_ohe.astype("float32"), pretrained_embeddings, transpose_b=True
)
quantized = tf.reshape(quantized, (-1, *((128, 64, 64, 32)[1:])))

# Generate images
decoder = vqvae_trainer.vqvae.get_layer("decoder")
generated_samples = decoder.predict(quantized)

for i in range(batch):
    plt.subplot(1, 2,  1)
    plt.imshow(priors[i])
    plt.title("Code")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(generated_samples[i].squeeze() + 0.5)
    plt.title("Generated Sample")
    plt.axis("off")
    plt.show()