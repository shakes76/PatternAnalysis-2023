from modules import *
from dataset import *
from matplotlib.pyplot	import plot, title, xlabel, ylabel, show, legend
from tensorflow import keras
from tensorflow.keras.optimizers import Adam 

def trained_vqvae():
    vqvae_model = VQVAE_MODEL(data_variance, 16, 128, 0.25)
    vqvae_model.compile(optimizer=Adam(), loss='mean_squared_error')
    vqvae_history = vqvae_model.fit(x_train_scaled, epochs = 50, batch_size=128)
    trained_model = vqvae_model.vqvae

    plot(vqvae_history.history["reconstruction_loss"])
    title("VQVAE Training Loss per Epoch")
    ylabel("Loss")
    xlabel("Epoch")
    show()

    return trained_model, vqvae

def train_pcnn(trained):
    encoder	= trained.vqvae.get_layer("encoder")
    quantizer = trained.vqvae.get_layer("quantizer")
    output_enco = encoder.predict(x_test_scaled)
    codebook = quantizer.code_indices(output_enco.reshape(-1, output_enco.shape[-1]))
    codebook = codebook.numpy().reshape(output_enco.shape[:-1])
    pcnn_model = pcnn()

def main():
    '''
    for training VQVAE and a PCNN
    '''
    vqvae, trained = trained_vqvae()
    vqvae.save("VQVAE.h5")
    train_pcnn(trained)

if __name__ == "__main__":
    main()