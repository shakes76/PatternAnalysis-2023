"""
modules.py

Author: Ethan Jones
Student ID: 44829531
COMP3710 OASIS brain StyleGAN project
Semester 2, 2023
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers

EPSILON = 0.00001
LATENT_DIMENSIONS = 512
ALPHA = 0.2
FILTERS = 8
SIZE = 256


def apply_noise(n_filters, image_size):
    """
    Apply noise to the input tensor

    param n_filters: Number of filters in the input tensor
    param image_size: Size of the image
    return: tensor with the noise added
    """
    input = layers.Input((image_size, image_size, n_filters))
    noise = layers.Input((image_size, image_size, 1))
    x = input

    # Add noise to the input
    x = x + noise
    return tf.keras.Model([input, noise], x)


def AdaIN(n_filters, image_size, epsilon=EPSILON):
    """
    Apply Adaptive Instance Normalisation to the input tensor

    param n_filters: The number of filters in the input tensor
    param image_size: The size of the input image
    param epsilon: The epsilon value to use in the calculation (GLOBAL VARIABLE
    return: The AdaIN model
    """
    input = layers.Input((image_size, image_size, n_filters))
    v = layers.Input(n_filters)
    x = input

    # Calculate scale and bias for AdaIN
    y_scale = layers.Dense(n_filters)(v)
    y_scale = layers.Reshape([1, 1, n_filters])(y_scale)
    y_bias = layers.Dense(n_filters)(v)
    y_bias = layers.Reshape([1, 1, n_filters])(y_bias)

    # Calculate mean and standard deviation of x
    dimensions = [1, 2]
    mean = tf.math.reduce_mean(x, dimensions, keepdims=True)
    stddev = tf.math.reduce_std(x, dimensions, keepdims=True) + epsilon

    # Apply AdaIN transformation to the tensor
    x = y_scale * ((x - mean) / (stddev + epsilon)) + y_bias
    return tf.keras.Model([input, v], x)


def WNetwork(lat_dim=LATENT_DIMENSIONS):
    """
    Maps the latent noise vector z to the style code vector v using a series of fully connected layers.

    param lat_dim: The number of dimensions in the latent space
    return: The Mapping Network transforming z to v
    """
    z = layers.Input([lat_dim])
    v = z
    for _ in range(8):
        v = layers.Dense(lat_dim)(v)
        v = layers.LeakyReLU(ALPHA)(v)
    return tf.keras.Model(z, v)


class Discriminator:

    def __init__(self):
        """
        Initialise the Discriminator network with the initial image size and number of filters
        """
        self.init_size = SIZE
        self.init_filters = FILTERS

    def discriminator_block(self, n_filters, image_size):
        """
        Constructs a single block of the Discriminator network

        param n_filters: Number of filters for the convolutional layers
        param image_size: Size of the input image for this block
        return: A Keras model representing the discriminator block
        """
        if image_size == self.init_size:
            input_tensor = layers.Input(shape=(image_size, image_size, 1))
        else:
            input_tensor = layers.Input(shape=(image_size, image_size, n_filters // 2))

        x = input_tensor
        x = layers.Conv2D(n_filters, kernel_size=3, padding="same")(x)
        x = layers.Conv2D(n_filters, kernel_size=3, padding="same")(x)
        x = layers.AveragePooling2D((2, 2))(x)
        x = layers.LeakyReLU(ALPHA)(x)
        return tf.keras.Model(input_tensor, x)

    def discriminator(self):
        """
        Constructs the full Discriminator network

        The network dynamically adjusts its depth based on the initial image size
        and ends with a dense layer with a sigmoid activation to output a probability

        return: Keras model representing the full discriminator network
        """
        current_size = self.init_size
        n_filters = self.init_filters
        input_tensor = layers.Input(shape=[current_size, current_size, 1])
        x = input_tensor

        while current_size > 4:
            x = self.discriminator_block(n_filters, current_size)(x)
            current_size = current_size // 2
            n_filters = 2 * n_filters

        x = layers.Conv2D(n_filters, kernel_size=3, padding="same")(x)
        x = layers.Conv2D(n_filters, kernel_size=3, padding="same")(x)
        x = layers.LeakyReLU(ALPHA)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(1, activation="sigmoid")(x)

        return tf.keras.Model(input_tensor, x)
