"""
Source code for training, validation, testing, and saving of the model
"""
from dataset import *
from modules import *

import sys
from os import mkdir
from os.path import exists

import keras.callbacks
from keras import optimizers, losses
import matplotlib.pyplot as plt
import pandas as pd

# Change as needed
FILE_PATH = '/home/groups/comp3710/ADNI/AD_NC/'
VQVAE_WEIGHTS_PATH = './VQVAEWeights/'
PIXEL_WEIGHTS_PATH = './PixelWeights/'
RESULTS_PATH = './Results/'
IMG_SHAPE = 256
SEED = 69
NUM_IMAGES_TO_SHOW = 5

# Toggle for training requirements.
TRAINING_VQVAE = True
TRAINING_PIXELCNN = True

# Hyper-parameters - tune as needed
RGB = True
BETA = 0.25
NUM_EMBEDDINGS = 256
BATCH_SIZE = 32
LATENT_DIM = 32
PIXEL_SHIFT = 0.5
NUM_EPOCHS_VQVAE = 7
NUM_EPOCHS_PIXEL = 30
VALIDATION_SPLIT = 0.3
VQVAE_LEARNING_RATE = 0.001
VQVAE_OPTIMISER = optimizers.Adam(learning_rate=VQVAE_LEARNING_RATE)
NUM_RESIDUAL_LAYERS = 2
NUM_PIXEL_B_LAYERS = 2
NUM_PIXEL_FILTERS = 128
PIXEL_KERNEL_SIZE = 7
PIXEL_ACTIVATION = 'relu'
PIXEL_CNN_LEARNING_RATE = VQVAE_LEARNING_RATE
PIXEL_CNN_OPTIMISER = VQVAE_OPTIMISER
PIXEL_CNN_LOSS = losses.SparseCategoricalCrossentropy(from_logits=True)

num_channels = 3 if RGB else 1

# GPU config
gpu_used = len(tf.config.list_physical_devices('GPU'))
device = '/GPU:0' if gpu_used else '/CPU:0'

train_data = prep_data(path=FILE_PATH + 'train', img_dim=IMG_SHAPE,
                       batch_size=BATCH_SIZE, RGB=RGB,
                       validation_split=VALIDATION_SPLIT,
                       subset='training', seed=SEED, shift=PIXEL_SHIFT)

val_data = prep_data(path=FILE_PATH + "train", img_dim=IMG_SHAPE,
                     batch_size=BATCH_SIZE, RGB=RGB,
                     validation_split=VALIDATION_SPLIT,
                     subset="validation", seed=SEED, shift=PIXEL_SHIFT)

test_data = prep_data(path=FILE_PATH + 'test', img_dim=IMG_SHAPE, batch_size=BATCH_SIZE, RGB=RGB, shift=PIXEL_SHIFT)

# Calculate training variance across pixels
full_train_data = train_data.unbatch()
num_train_images = full_train_data.reduce(np.int32(0), lambda x, _: x + 1).numpy()

full_val_data = val_data.unbatch()
num_val_images = full_val_data.reduce(np.int32(0), lambda x, _: x + 1).numpy()

full_test_data = val_data.unbatch()
num_test_images = full_test_data.reduce(np.int32(0), lambda x, _: x + 1).numpy()

# Calculate training variance
num_train_pixels = num_train_images * num_channels * IMG_SHAPE ** 2

image_sum = full_train_data.reduce(np.float32(0), lambda x, y: x + y).numpy().flatten()
tot_pixel_sum = image_sum.sum()
mean_pixel_train = tot_pixel_sum / num_train_pixels

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
vqvae = VQVAE(tr_var=var_pixel_train, num_encoded=NUM_EMBEDDINGS, latent_dim=LATENT_DIM,
              beta=BETA, num_channels=num_channels)
# Create directories if they do not already exist
if not exists(VQVAE_WEIGHTS_PATH):
    mkdir(VQVAE_WEIGHTS_PATH)

if not exists(RESULTS_PATH):
    mkdir(RESULTS_PATH)

# Store training loss/metrix on CSV
training_csv_logger = keras.callbacks.CSVLogger(VQVAE_WEIGHTS_PATH + 'training.log', separator=',', append=False)

final_train_mean_ssim = final_val_mean_ssim = 0

# Train the VQVAE
vqvae.compile(optimizer=VQVAE_OPTIMISER)

print('Training beginning ')
if TRAINING_VQVAE:
    with tf.device(device):
        vqvae.fit(train_data, epochs=NUM_EPOCHS_VQVAE, callbacks=[training_csv_logger], validation_data=val_data)

        # Final SSIM Values
        final_train_mean_ssim = mean_ssim(train_data, num_train_images, vqvae)
        final_val_mean_ssim = mean_ssim(val_data, num_val_images, vqvae)

        # Print results to file
        main_stdout = sys.stdout

        with open(RESULTS_PATH + 'main_results.txt', 'a') as f:
            sys.stdout = f
            print(f'Final training mean ssim {final_train_mean_ssim}')
            print(f'Final validation mean ssim {final_val_mean_ssim}')
            sys.stdout = main_stdout

        # Save the trained model
        vqvae.save_weights(VQVAE_WEIGHTS_PATH + 'trained_model_weights')

# Load the trained model
vqvae = VQVAE(tr_var=var_pixel_train, num_encoded=NUM_EMBEDDINGS, latent_dim=LATENT_DIM, num_channels=num_channels)
vqvae.load_weights(VQVAE_WEIGHTS_PATH + 'trained_model_weights')


def plot_results(epoch_results):
    """ Plots and saves all train/val losses"""
    # Total losses
    plt.figure()
    plt.plot(epoch_results['total_loss'], label='Total Loss (Training)')
    plt.plot(epoch_results['val_total_loss'], label='Total Loss (Validation)')

    plt.xlabel('Epoch Num')
    plt.ylabel('Loss')
    plt.title('VQVAE Total Loss')
    plt.legend(loc='upper right')

    plt.savefig(RESULTS_PATH + 'tot_loss.png')

    # Reconstruction losses
    plt.figure()
    plt.plot(epoch_results['reconstruction_loss'], label='Reconstruction Loss (Training)')
    plt.plot(epoch_results['val_reconstruction_loss'], label='Reconstruction Loss (Validation)')

    plt.xlabel('Epoch Num')
    plt.ylabel('Loss')
    plt.title('VQVAE Reconstruction Loss')
    plt.legend(loc='upper right')

    plt.savefig(RESULTS_PATH + 'recon_loss.png')

    # Quantisation losses
    plt.figure()
    plt.plot(epoch_results['vq_loss'], label='Quantisation Loss (Training)')
    plt.plot(epoch_results['val_vq_loss'], label='Quantisation Loss (Validation)')

    plt.xlabel('Epoch Num')
    plt.ylabel('Loss')
    plt.title('Quantisation Loss')
    plt.legend(loc='upper right')

    plt.savefig(RESULTS_PATH + 'quant_loss.png')


training_results = pd.read_csv(VQVAE_WEIGHTS_PATH + 'training.log', sep=',', engine='python')
plot_results(training_results)


# Print overall mean test SSIM for VQ-VAE:
with tf.device(device):
    test_mean_ssim_vqvae = mean_ssim(test_data, num_test_images, vqvae)
    main_stdout = sys.stdout
    with open(RESULTS_PATH + 'main_results.txt', 'a') as f:
        sys.stdout = f
        print("VQ-VAE Test Mean SSIM:", test_mean_ssim_vqvae)
        sys.stdout = main_stdout


def get_codebook_indices_wrapper(encoder, vq):
    """
    Wrapper function to mitigate memory issues
    :param encoder: Encoder portion of vqvae
    :param vq: The vq layer of vqvae
    :return: mapper() object
    """
    def mapper(data):
        """
        The mapper to be used with tensorflow dset
        """
        encoded = encoder(data)
        encoded_flattened = tf.reshape(encoded, [-1, tf.shape(encoded)[-1]])
        codebook_indices_flattened = vq.get_codebook_indices(encoded_flattened)
        codebook_indices = tf.reshape(codebook_indices_flattened, tg.shape(encoded)[:-1])
        return codebook_indices

    return mapper


# Get learned VQ-VAE codebooks
learned_codebook_indices = get_codebook_indices_wrapper(vqvae.get_encoder(), vqvae.get_vq())
codebook_train = train_data.map(learned_codebook_indices)
codebook_val = val_data.map(learned_codebook_indices)

# Create PixelCNN model
pixelCNN = PixelCNN(num_res=NUM_RESIDUAL_LAYERS, num_pixel_B=NUM_PIXEL_B_LAYERS,
                    num_encoded=NUM_EMBEDDINGS, num_filters=NUM_PIXEL_FILTERS,
                    kernel_size=PIXEL_KERNEL_SIZE, activation=PIXEL_ACTIVATION)
pixelCNN.compile(optimizer=PIXEL_CNN_OPTIMISER, loss=PIXEL_CNN_LOSS)

# Create directory to store model weights (if said directory does not already exist)
if not exists(PIXEL_WEIGHTS_PATH):
    mkdir(PIXEL_WEIGHTS_PATH)

# Store training loss/metric results in CSV file - change append parameter to True if wanting to
# continue training
training_csv_logger_pixel = keras.callbacks.CSVLogger(PIXEL_WEIGHTS_PATH + 'training_pixel.log',
                                                      separator=',', append=False)

# Train PixelCNN (with trained VQ-VAE codebooks)
if TRAINING_PIXELCNN:
    with tf.device(device):
        pixelCNN.fit(codebook_train, epochs=NUM_EPOCHS_PIXEL, callbacks=[training_csv_logger_pixel],
                     validation_data=codebook_val)
        # Save trained model
        pixelCNN.save_weights(PIXEL_WEIGHTS_PATH + "trained_model_weights")

# Load trained model
pixelCNN = PixelCNN(num_res=NUM_RESIDUAL_LAYERS, num_pixel_B=NUM_PIXEL_B_LAYERS,
                    num_encoded=NUM_EMBEDDINGS, num_filters=NUM_PIXEL_FILTERS,
                    kernel_size=PIXEL_KERNEL_SIZE, activation=PIXEL_ACTIVATION)
pixelCNN.load_weights(PIXEL_WEIGHTS_PATH + "trained_model_weights")


# Plot losses
def plot_pixel_train_val_results(epoch_results):
    """
    Plots and saves the train/val losses of the PixelCNN
    FLAG: I actually did not end up using these plots, and instead created my own using training.log fata
    """
    # Total losses
    plt.figure()
    plt.plot(epoch_results['loss'], label='Training loss')
    plt.plot(epoch_results['val_loss'], label='Validation loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('(Sparse) Categorical Cross Entropy Loss')
    plt.legend(loc='upper right')

    plt.savefig(RESULTS_PATH + 'pixel_losses.png')


training_results_pixel = pd.read_csv(PIXEL_WEIGHTS_PATH + 'training_pixel.log', sep=',',
                                     engine='python')
plot_pixel_train_val_results(training_results_pixel)
