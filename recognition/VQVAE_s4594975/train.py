from modules import *
from dataset import *
from matplotlib.pyplot	import plot, title, xlabel, ylabel, show, legend
from tensorflow import keras
from tensorflow.keras.optimizers import Adam 

def trained_vqvae():
    """
    Train a VQ-VAE model and return the trained model.

    Returns:
        trained_model (keras.Model): The trained VQ-VAE model.
        vqvae_model (keras.Model): The VQ-VAE model including encoder, quantizer, and decoder.
    """
    vqvae_model = VQVAE_MODEL(data_variance, 16, 128, 0.25)
    vqvae_model.compile(optimizer=Adam(), loss='mean_squared_error')
    vqvae_history = vqvae_model.fit(x_train_scaled, epochs = 20, batch_size=128)
    trained_model = vqvae_model.vqvae

    plot(vqvae_history.history["reconstruction_loss"])
    title("VQVAE Training Loss per Epoch")
    ylabel("Loss")
    xlabel("Epoch")
    show()
    return trained_model, vqvae_model

def train_pcnn(trained):
    """
    Train a Pixel-CNN model using a trained VQ-VAE as a prior model.

    Args:
        trained (keras.Model): The trained VQ-VAE model.

    Returns:
        None
    """
    encoder	= trained.vqvae.get_layer("encoder")
    quantizer = trained.vqvae.get_layer("quantizer")
    output_enco = encoder.predict(x_test_scaled)
    codebook = quantizer.get_code_indices(output_enco.reshape(-1, output_enco.shape[-1]))
    codebook = codebook.numpy().reshape(output_enco.shape[:-1])
    pcnn_model = pcnn(trained, output_enco)
    pcnn_model.compile(optimizer=keras.optimizers.Adam(3e-4), 
                       loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"],)

    pixelcnn_history = pcnn_model.fit(x=codebook, y=codebook, batch_size=128, epochs=20, validation_split=0.3,)
    pcnn_model.save("PCNN.h5")
    plt.plot(pixelcnn_history.history['loss'])
    plt.plot(pixelcnn_history.history['val_loss'])
    plt.title('Pixel-CNN Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def main():
    """
    Main function for training VQ-VAE and Pixel-CNN models.
    """
    vqvae, trained = trained_vqvae()
    vqvae.save("VQVAE.h5")
    train_pcnn(trained)

if __name__ == "__main__":
    main()