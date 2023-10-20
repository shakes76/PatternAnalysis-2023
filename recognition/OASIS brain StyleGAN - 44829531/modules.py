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
    scaled_noise = layers.Dense(n_filters)(noise)
    x = x + scaled_noise
    return tf.keras.Model([input, noise], x)


def AdaIN(n_filters, image_size, epsilon=0.00000001):
    """
    Apply Adaptive Instance Normalisation to the input tensor

    param n_filters: The number of filters in the input tensor
    param image_size: The size of the input image
    param epsilon: The epsilon value to use in the calculation (GLOBAL VARIABLE
    return: The AdaIN model
    """
    input = layers.Input((image_size, image_size, n_filters))
    w = layers.Input(512)
    x = input

    # Calculate scale and bias for AdaIN
    y_scale = layers.Dense(n_filters)(w)
    y_scale = layers.Reshape([1, 1, n_filters])(y_scale)
    y_bias = layers.Dense(n_filters)(w)
    y_bias = layers.Reshape([1, 1, n_filters])(y_bias)

    # Calculate mean and standard deviation of x
    dimensions = [1, 2]
    mean = tf.math.reduce_mean(x, dimensions, keepdims=True)
    stddev = tf.math.reduce_std(x, dimensions, keepdims=True) + epsilon

    # Apply AdaIN transformation to the tensor
    x = y_scale * ((x - mean) / (stddev + epsilon))
    return tf.keras.Model([input, w], x)


def WNetwork(lat_dim=512):
    """
    Maps the latent noise vector z to the style code vector v using a
    series of fully connected layers.

    param lat_dim: The number of dimensions in the latent space
    return: The Mapping Network transforming z to w
    """
    z = layers.Input(shape=[lat_dim])
    w = z
    for _ in range(8):
        w = layers.Dense(lat_dim)(w)
        w = layers.LeakyReLU(0.2)(w)
    return tf.keras.Model(z, w)


class Generator:
    """
    The generator network for the StyleGAN model.

    The generator is trained to fool discriminator by generating fake
    distribution which is close to the true distribution
    """
    def __init__(self):
        self.init_size = 4
        self.init_filters = 512

    def generator_block(self, n_filters, image_size):
        """
        Create a generator block for the StyleGAN model

        param n_filters: The number of filters for the convolutional layers
        param image_size: size of the image
        return: A keras model for the generator block
        """

        # Define the input tensors
        input_tensor = layers.Input(shape=(image_size, image_size, n_filters))
        noise = layers.Input(shape=(2 * image_size, 2 * image_size, 1))
        w = layers.Input(shape=512)
        x = input_tensor

        # Halve the number of filters for the next block
        n_filters = n_filters // 2

        # Upsample the input tensor
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)

        # Apply two convolutional layers with noise and AdaIN normalisation
        x = layers.Conv2D(n_filters, kernel_size=3, padding="same")(x)
        x = apply_noise(n_filters, image_size * 2)([x, noise])
        x = AdaIN(n_filters, image_size * 2)([x, w])
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = layers.Conv2D(n_filters, kernel_size=3, padding="same")(x)
        x = apply_noise(n_filters, image_size * 2)([x, noise])
        x = AdaIN(n_filters, image_size * 2)([x, w])
        x = layers.LeakyReLU(alpha=0.2)(x)

        return tf.keras.Model([input_tensor, w, noise], x)

    def generator(self):
        """
        Define the overall generator network for the StyleGAN model

        return: A keras model for the generator network
        """

        current_size = self.init_size
        n_filters = self.init_filters
        input = layers.Input(shape=(current_size, current_size, n_filters))
        x = input
        i = 0

        # List of noise inputs and latent vector inputs
        noise_inputs, z_inputs = [], []
        curr_size = self.init_size

        # Create the noise and latent vector inputs for each block
        while curr_size <= 256:
            noise_inputs.append(layers.Input(shape=[curr_size, curr_size, 1]))
            z_inputs.append(layers.Input(shape=[512]))
            curr_size *= 2

        # Create the mapping network
        mapping = WNetwork()

        # Apply the initial convolutional layer
        x = layers.Activation("linear")(x)
        x = apply_noise(n_filters, current_size)([x, noise_inputs[i]])
        x = AdaIN(n_filters, current_size)([x, mapping(z_inputs[i])])
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = layers.Conv2D(512, kernel_size=3, padding="same")(x)
        x = apply_noise(n_filters, current_size)([x, noise_inputs[i]])
        x = AdaIN(n_filters, current_size)([x, mapping(z_inputs[i])])
        x = layers.LeakyReLU(alpha=0.2)(x)

        # Apply the generator blocks until desired image size is reached
        while current_size < 256:
            i += 1
            x = self.generator_block(n_filters, current_size)([x, mapping(z_inputs[i]), noise_inputs[i]])
            current_size = 2 * current_size
            n_filters = n_filters // 2

        # Apply the final convolutional layer
        x = layers.Conv2D(1, kernel_size=3, padding="same", activation="sigmoid")(x)
        return tf.keras.Model([input, z_inputs, noise_inputs], x)


class Discriminator:
    """
    The Discriminator is trained to distinguish true and fake distribution
    as well as possible
    """
    def __init__(self):
        """
        Initialise the Discriminator network with the initial image size and
        number of filters
        """
        self.init_size = 256
        self.init_filters = 8

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
            input_tensor = layers.Input(shape=(image_size, image_size,
                                               n_filters // 2))

        x = input_tensor
        x = layers.Conv2D(n_filters, kernel_size=3, padding="same")(x)
        x = layers.Conv2D(n_filters, kernel_size=3, padding="same")(x)
        x = layers.AveragePooling2D((2, 2))(x)
        x = layers.LeakyReLU(0.2)(x)
        
        return tf.keras.Model(input_tensor, x)

    def discriminator(self):
        """
        Constructs the full Discriminator network

        The network dynamically adjusts its depth based on the
        initial image size and ends with a dense layer with a sigmoid
        activation to output a probability

        return: Keras model representing the full discriminator network
        """
        current_size = self.init_size
        n_filters = self.init_filters
        input_tensor = layers.Input(shape=[current_size, current_size, 1])
        x = input_tensor

        # Apply the discriminator blocks until the image size is 4x4
        while current_size > 4:
            x = self.discriminator_block(n_filters, current_size)(x)
            current_size = current_size // 2
            n_filters = 2 * n_filters

        # Apply two convolutional layers with leaky ReLU activation
        x = layers.Conv2D(n_filters, kernel_size=3, padding="same")(x)
        x = layers.Conv2D(n_filters, kernel_size=3, padding="same")(x)
        x = layers.LeakyReLU(0.2)(x)

        # Flatten the tensor and pass through a dense layer with
        # sigmoid activation
        x = layers.Flatten()(x)
        x = layers.Dense(1, activation="sigmoid")(x)

        return tf.keras.Model(input_tensor, x)
