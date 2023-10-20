import tensorflow as tf
import random
from tensorflow import keras
from modules import VQVAETrainer, get_pixel_cnn
from dataset import get_train_dataset, get_test_dataset, get_dataset_variance
from matplotlib import pyplot as plt

def plot_vqvae_history(history):
    """
    Plot the loss and ssim of  VQ-VAE trainer
    """
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

train_ds = get_train_dataset()
train_variance = get_dataset_variance(train_ds)

def train_vqvae():
    '''Train the VQ-VAE model'''
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
    vqvae_history = vqvae_trainer.fit(train_ds, epochs=50, callbacks=[checkpoint])
    # Plot and save the training history
    plot_vqvae_history(vqvae_history)
    
def visualize_vqvae_results():
    """
    Generate reconstructed images using trained VQ-VAE
    """
    # Load the trained weights to vq-vae trainer
    vqvae_trainer = VQVAETrainer(0.03525, latent_dim=32, num_embeddings=128)
    vqvae_trainer.load_weights('recognition/VQ-VAE-46495408/checkpoint/vqvae_ckpt')
    
    trained_vqvae_model = vqvae_trainer.vqvae
    encoder = vqvae_trainer.vqvae.get_layer("encoder")
    quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")
    
    test_images = get_test_dataset().take(1)
    test_batch = next(iter(test_images))
    # Get the reconstruction results
    reconstructions_test = trained_vqvae_model.predict(test_images)
    rec_batch = reconstructions_test
    # Get the codebook indice
    encoded_outputs = encoder.predict(test_images)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
    codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
    
    n = 1
    num_samples = 10
    plt.figure(figsize=(10, 30))
    for i in random.sample(range(0, 127), num_samples):
        ssim = tf.image.ssim(test_batch[i], rec_batch[i], max_val=1.0)
        # Plot the original image
        plt.subplot(num_samples, 3, n)
        plt.imshow(test_batch[i])
        plt.title("Original")
        plt.axis("off")
        
        # Plot the discrete code
        plt.subplot(num_samples, 3, n + 1)
        plt.imshow(codebook_indices[i])
        plt.title("SSIM: {:.2f} \nCode".format(ssim))
        plt.axis("off")
        
        # Plot the reconstructed image
        plt.subplot(num_samples, 3, n + 2)
        plt.imshow(rec_batch[i])
        plt.title("Reconstructed")
        plt.axis("off")
        n += 3
        
    plt.tight_layout()
    plt.savefig('recognition/VQ-VAE-46495408/results/vqvae_test_images.png')
    plt.show()
    
def plot_pixelcnn_history(history):
    """
    Plot the training results of Pixel CNN model
    """
    # Plot the loss for PixelCNN model
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Pixel CNN Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.savefig('recognition/VQ-VAE-46495408/results/pixelcnn_loss.png')
    plt.close()
    
    # Plot the accuracy for PixelCNN model
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Pixel CNN Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.savefig('recognition/VQ-VAE-46495408/results/pixelcnn_accuracy.png')
    plt.close()
    
def train_pixelcnn():
    """Train the Pixel CNN Model"""
    # Load the trained weights to vq-vae trainer
    vqvae_trainer = VQVAETrainer(train_variance, latent_dim=32, num_embeddings=128)
    vqvae_trainer.load_weights('recognition/VQ-VAE-46495408/checkpoint/vqvae_ckpt')
    encoder = vqvae_trainer.vqvae.get_layer("encoder")
    quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")
    
    # Generate the codebook indices
    encoded_outputs = encoder.predict(train_ds)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
    codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
    print(f"Shape of the training data for PixelCNN: {codebook_indices.shape}")
    
    # Compile and train
    pixelcnn_input_shape = encoded_outputs.shape[1:-1]
    pixel_cnn = get_pixel_cnn(pixelcnn_input_shape, vqvae_trainer.num_embeddings)
    pixel_cnn.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    
    # Create a callback that saves the model's weights
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "recognition/VQ-VAE-46495408/checkpoint/pixelcnn_ckpt",
        save_best_only=False,
        save_weights_only=True
    )
    
    pixelcnn_history = pixel_cnn.fit(
        x=codebook_indices,
        y=codebook_indices,
        epochs=800,
        validation_split=0.1,
        callbacks=[checkpoint]
    )
    # Plot and save the training history
    plot_pixelcnn_history(pixelcnn_history)
    
if __name__=='__main__':
    train_vqvae()
    train_pixelcnn()