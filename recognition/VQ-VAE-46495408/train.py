import tensorflow as tf
from tensorflow import keras
from modules import VQVAETrainer
from dataset import get_train_dataset, get_dataset_variance

'''Train the VQ-VAE model'''
train_ds = get_train_dataset()
train_variance = get_dataset_variance(train_ds)
vqvae_trainer = VQVAETrainer(train_variance, latent_dim=32, num_embeddings=128)
# Define the optimizer
optimizer = keras.optimizers.Adam()
vqvae_trainer.compile(optimizer=optimizer)
# Create a callback that saves the model's weights
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "recognition/VQ-VAE-46495408",
    monitor='ssim',
    save_weights_only = True,
    save_freq='epoch'
)
vqvae_trainer.fit(train_ds, epochs=2, callbacks=[checkpoint])