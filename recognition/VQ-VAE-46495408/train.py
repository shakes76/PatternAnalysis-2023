import tensorflow as tf
from tensorflow import keras
from modules import VQVAETrainer
from dataset import get_train_dataset, get_dataset_variance, get_test_dataset
from matplotlib import pyplot as plt

def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['reconstruction_loss'])
    plt.plot(history.history['vqvae_loss'])
    plt.title('VQ-VAE training loss')
    plt.xlabel('epoch')
    plt.legend(['total loss', 'reconstruction loss', 'vqvae loss'], loc='upper right')
    plt.savefig('recognition/VQ-VAE-46495408/results/vqvae_training_loss.png')
    plt.close()
    
    plt.plot(history.history['ssim'])
    plt.title('VQ-VAE ssim')
    plt.xlabel('epoch')
    plt.ylabel('ssim')
    plt.savefig('recognition/VQ-VAE-46495408/results/vqvae_ssim.png')
    plt.close()

'''Train the VQ-VAE model'''
train_ds = get_test_dataset()
def train_vqvae():
    train_variance = get_dataset_variance(train_ds)
    vqvae_trainer = VQVAETrainer(train_variance, latent_dim=32, num_embeddings=128)
    # Define the optimizer
    optimizer = keras.optimizers.Adam()
    vqvae_trainer.compile(optimizer=optimizer)
    # Create a callback that saves the model's weights
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "recognition/VQ-VAE-46495408/checkpoint/vqvae_ckpt",
        save_best_only=False,
        save_weights_only=True
    )
    vqvae_history = vqvae_trainer.fit(train_ds, epochs=1, callbacks=[checkpoint])
    # Plot the training history
    plot_history(vqvae_history)
    
if __name__=='__main__':
    train_vqvae()