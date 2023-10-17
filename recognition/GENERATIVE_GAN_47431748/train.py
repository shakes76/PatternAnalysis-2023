"""
Source code for training, validation, testing, and saving of the model
"""
from dataset import *
from modules import *

import keras.callbacks
from keras import optimizers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

# Change as needed
FILE_PATH = r'C:/Users/jackc/OneDrive/Desktop/UQ/UQ23S2/COMP3710_PROJ/PatternAnalysis-2023/AD_NC/'
IMG_DIM = 256
SEED = 69

# Hyper-parameters - tune as needed
NUM_EMBEDDINGS = 256
BATCH_SIZE = 32
LATENT_DIM = 32
PIXEL_SHIFT = 0.5
NUM_EPOCHS = 30
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
num_train_pixels = num_train_images * IMG_DIM ** 2

image_sum = full_train_data.reduce(np.float32(0), lambda x, y: x + y).numpy().flatten()
tot_pixel_sum = image_sum.sum()
mean_pixel_train = tot_pixel_sum / num_train_pixels

# Calculate training variance
image_sse = full_train_data.reduce(np.float32(0), lambda x, y: x + (y - mean_pixel_train) ** 2).numpy().flatten()
var_pixel_train = image_sse.sum() / (num_train_pixels - 1)

# VQVAE
vqvae = VQVAE(tr_var=var_pixel_train, num_encoded=NUM_EMBEDDINGS, latent_dim=LATENT_DIM)

# Define SSIM metric + callback
def mean_ssim(data, model):
    """
    Calculate mean SSUM over dataset
    :param data: The given dataset
    :param model: The model
    :return: Mean SSIM over data
    """
    ssim_sum = 0
    num_images = 0

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
        num_images += image_batch.shape[0]
        print(f'Number of images in batch is: {num_images}')

    return ssim_sum / num_images

class SSIMTracking(keras.callbacks.Callback):
    """
    Calculates + displats mean SSIM each epoch
    """
    def __init__(self, dataset, dataset_name='Training'):
        super(SSIMTracking, self).__init__()
        self._dataset = dataset
        self._dataset_name = dataset_name

    def on_epoch_end(self, epoch, logs=None):
        """
        Calculate + display mean SSIM on each epoch
        :param epoch: Epoch number
        :param logs:
        :return: None
        """
        print(f'Epoch {epoch}: {self._dataset_name} mean SSIM: {mean_ssim(self._dataset, self.model)}')

# Train the VQVAE
vqvae.compile(optimizer=OPTIMISER)

with tf.device(device):
    history = vqvae.fit(
        train_data,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[SSIMTracking(dataset=train_data),
                   SSIMTracking(val_data, dataset_name='Validation')]
    )