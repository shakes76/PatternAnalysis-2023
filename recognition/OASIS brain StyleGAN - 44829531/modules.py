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


