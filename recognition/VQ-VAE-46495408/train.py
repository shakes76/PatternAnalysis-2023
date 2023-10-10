import tensorflow as tf
from tensorflow import keras
from modules import VQVAETrainer, get_pixel_cnn
from dataset import get_train_dataset, get_dataset_variance, get_validate_dataset, get_test_dataset
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

train_ds = get_test_dataset()

def train_vqvae():
    '''Train the VQ-VAE model'''
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
    vqvae_history = vqvae_trainer.fit(train_ds, epochs=5, callbacks=[checkpoint])
    # Plot and save the training history
    plot_history(vqvae_history)
    
def train_pixelcnn():
    # Load the trained weights to vq-vae trainer
    train_variance = get_dataset_variance(train_ds)
    vqvae_trainer = VQVAETrainer(train_variance, latent_dim=32, num_embeddings=128)
    vqvae_trainer.load_weights('recognition/VQ-VAE-46495408/checkpoint/vqvae_ckpt')
    encoder = vqvae_trainer.vqvae.get_layer("encoder")
    quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")
    # Generate the codebook indices
    encoded_outputs = encoder.predict(train_ds)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
    codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
    #print(f"Shape of the training data for PixelCNN: {codebook_indices.shape}")
    # Compile and train
    pixelcnn_input_shape = encoded_outputs.shape[1:-1]
    pixel_cnn = get_pixel_cnn(pixelcnn_input_shape, vqvae_trainer.num_embeddings)
    pixel_cnn.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    pixel_cnn.fit(
        x=codebook_indices,
        y=codebook_indices,
        epochs=5,
        validation_split=0.1,
    )
    
if __name__=='__main__':
    train_pixelcnn()