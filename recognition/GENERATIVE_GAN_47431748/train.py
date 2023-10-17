"""
Source code for training, validation, testing, and saving of the model
"""
from dataset import *
from modules import *

from os import mkdir
from os.path import exists
import keras.callbacks
from keras import optimizers, losses
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import pandas as pd

# Change as needed
FILE_PATH = r'C:/Users/jackc/OneDrive/Desktop/UQ/UQ23S2/COMP3710_PROJ/PatternAnalysis-2023/AD_NC/'
SAVED_WEIGHTS_PATH = './SavedWeights/'
RESULTS_PATH = './Results/'
IMG_DIM = 256
SEED = 69
NUM_IMAGES_TO_SHOW = 10

TRAINING_VQVAE = False  # Set to True 4 training
TRAINING_PIXELCNN = True # Same as above

# Hyper-parameters - tune as needed
NUM_EMBEDDINGS = 256
BATCH_SIZE = 32
LATENT_DIM = 32
PIXEL_SHIFT = 0.5
NUM_EPOCHS_VQVAE = 7
NUM_EPOCHS_PIXEL = 30
VALIDATION_SPLIT = 0.3
LEARNING_RATE = 0.001
OPTIMISER = optimizers.Adam(learning_rate=LEARNING_RATE)

# GPU config
gpu_used = len(tf.config.list_physical_devices('GPU'))
device = '/GPU:0' if gpu_used else '/CPU:0'

train_data = prep_data(path=FILE_PATH + 'train', img_dim=IMG_DIM,
                       batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT,
                       subset='training', seed=SEED, shift=PIXEL_SHIFT)

val_data = prep_data(path=FILE_PATH + "train", img_dim=IMG_DIMENSION,
                                      batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT,
                                      subset="validation", seed=SEED, shift=PIXEL_SHIFT)

test_data = prep_data(path=FILE_PATH + 'test', img_dim=IMG_DIM, batch_size=BATCH_SIZE, shift=PIXEL_SHIFT)

# Calculate training variance across pixels
full_train_data = train_data.unbatch()
num_train_images = full_train_data.reduce(np.int32(0), lambda x, _: x + 1).numpy()

full_val_data = val_data.unbatch()
num_val_images = full_val_data.reduce(np.int32(0), lambda x, _: x + 1).numpy()

full_test_data = val_data.unbatch()
num_test_images = full_test_data.reduce(np.int32(0), lambda x, _: x + 1).numpy()

# Calculate training variance
num_train_pixels = num_train_images * IMG_DIM ** 2

image_sum = full_train_data.reduce(np.float32(0), lambda x, y: x + y).numpy().flatten()
tot_pixel_sum = image_sum.sum()
mean_pixel_train = tot_pixel_sum / num_train_pixels

# Calculate training variance
image_sse = full_train_data.reduce(np.float32(0), lambda x, y: x + (y - mean_pixel_train) ** 2).numpy().flatten()
var_pixel_train = image_sse.sum() / (num_train_pixels - 1)


# Define SSIM metric
def mean_ssim(data, data_size, model):
    """
    Calculate mean SSUM over dataset
    :param data: The given dataset
    :param data_size: The number of elements in the dset
    :param model: The model
    :return: Mean SSIM over data
    """
    ssim_sum = 0

    for image_batch in data:
        # Unshift normalised images + reconstructions
        image_batch_unshifted = image_batch + PIXEL_SHIFT
        reconstruction_batch = model.predict(image_batch)
        reconstruction_batch_unshifted = reconstruction_batch + PIXEL_SHIFT

        # Sum over all SSUMs
        total_batch_ssim = tf.math.reduce_sum(tf.image.ssim(image_batch_unshifted,
                                                            reconstruction_batch_unshifted,
                                                            max_val=1.0))

        # Add batch to total
        ssim_sum += total_batch_ssim

    return (ssim_sum / data_size).numpy()

# VQ-VAE Model
vqvae = VQVAE(tr_var=var_pixel_train, num_encoded=NUM_EMBEDDINGS, latent_dim=LATENT_DIM)


if not exists(SAVED_WEIGHTS_PATH):
    mkdir(SAVED_WEIGHTS_PATH)

# Store training loss/metrix on CSV
training_csv_logger = keras.callbacks.CSVLogger(SAVED_WEIGHTS_PATH + 'training.log', separator=',', append=False)

final_train_mean_ssim = final_val_mean_ssim = 0

# Train the VQVAE
vqvae.compile(optimizer=OPTIMISER)

if TRAINING_VQVAE:
    with tf.device(device):
        vqvae.fit(train_data, epochs=NUM_EPOCHS_VQVAE, callbacks=[training_csv_logger], validation=val_data)

        # Final SSIM Values
        final_train_mean_ssim = mean_ssim(train_data, num_train_images, vqvae)
        final_val_mean_ssim = mean_ssim(val_data, num_val_images, vqvae)
        print(f'Final Training Mean SSIM: {final_train_mean_ssim}')
        print(f'Final Validation Mean SSIM: {final_val_mean_ssim}')

        # Save the trained model
        vqvae.save(SAVED_WEIGHTS_PATH + 'trained_model_weights')

# Load the trained model
vqvae = VQVAE(tr_var=var_pixel_train, num_encoded=NUM_EMBEDDINGS, latent_dim=LATENT_DIM)
vqvae.load_weights(SAVED_WEIGHTS_PATH + 'trained_model_weights')


def plot_results(epoch_results):
    """ Plots and saves all train/val losses"""
    # Make folder to save plots
    if not exists(RESULTS_PATH):
        mkdir(RESULTS_PATH)

    # Total losses
    plt.figure()
    plt.plot(epoch_results['total_loss'], label='Total Loss (Training)')
    plt.plot(epoch_results['val_total_loss'], label='Total Loss (Validation)')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss')
    plt.legend(loc='upper right')

    plt.savefig(RESULTS_PATH + 'total_losses.png')

    # Reconstruction losses
    plt.figure()
    plt.plot(epoch_results['reconstruction_loss'], label='Reconstruction Loss (Training)')
    plt.plot(epoch_results['val_reconstruction_loss'], label='Reconstruction Loss (Validation)')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss')
    plt.legend(loc='upper right')

    plt.savefig(RESULTS_PATH + 'reconstruction_losses.png')

    # Quantisation losses
    plt.figure()
    plt.plot(epoch_results['vq_loss'], label='Quantisation Loss (Training)')
    plt.plot(epoch_results['val_vq_loss'], label='Quantisation Loss (Validation)')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss')
    plt.legend(loc='upper right')

    plt.savefig(RESULTS_PATH + 'quantisation_losses.png')


training_results = pd.read_csv(SAVED_WEIGHTS_PATH + 'training.log', sep=',', engine='python')
plot_results(training_results)

# Generate 10 reconstructions
for sample_batch in test_data.take(1).as_numpy_iterator():
    sample_batch = sample_batch[:NUM_IMAGES_TO_SHOW]

# Reconstruction
reconstructed = vqvae.predict(sample_batch)

# Code (Flattened)
encoder_outputs = vqvae.get_encoder().predict(sample_batch)
encoder_outputs_flattened = encoder_outputs.reshape(-1, encoder_outputs.shape[-1])
codebook_indices = vqvae.get_vq().get_codebook_indices(encoder_outputs_flattened)

# Code (reshaped)
codebook_indices = codebook_indices.numpy().reshape(encoder_outputs.shape[:-1])

# Unshift and reshape reconstructions and original images and print SSIM values
plt.figure()
for i in range(len(sample_batch)):
    test_image = tf.reshape(sample_batch[i], (1, IMG_DIMENSION, IMG_DIMENSION, num_channels))\
                     + PIXEL_SHIFT

    reconstructed_image = tf.reshape(reconstructed[i],
                                     (1, IMG_DIMENSION, IMG_DIMENSION, num_channels)) + PIXEL_SHIFT

    codebook_image = codebook_indices[i]

    plt.subplot(1, 3, 1)
    plt.imshow(tf.squeeze(test_image))
    plt.title("Test Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(codebook_image)
    plt.title("Codebook")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(tf.squeeze(reconstructed_image))
    plt.title("Reconstruction")
    plt.axis("off")

    plt.show()
    plt.savefig(RESULTS_PATH + f'vq_vae_reconstructions_{i}.png')

    ssim = tf.math.reduce_sum(tf.image.ssim(test_image, reconstructed_image, max_val=1.0)).numpy()
    print("SSIM between Test Image and Reconstruction: ", ssim)

# Print overall mean test SSIM for VQ-VAE:
with tf.device(device):
    test_mean_ssim_vqvae = mean_ssim(test_data, num_test_images, vqvae)
    print("VQ-VAE Test Mean SSIM:", test_mean_ssim_vqvae)
