"""
modules.py
Author: Francesca Brzoskowski
s4705512
Containing the source code of the components of your model. Each component is
implementated as a class or a function
Based on implementation from: https://keras.io/examples/generative/vq_vae/
"""
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf
image_size = (128, 128)

# Encoding Layer
class VectorQuantizer(layers.Layer):
    """
    A class representing the Vector Quantization layer.

    ...

    Attributes
    ----------
    num_embeddings : int
        Number of embedding vectors.
    embedding_dim : int
        Dimensionality of each embedding vector.
    beta : float
        Hyperparameter to control the strength of the commitment loss.

    Methods
    -------
    call(x)
        Performs the forward pass of the layer.
    get_code_indices(flattened_inputs)
        Calculates the indices of the closest embedding vectors.
    """

    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor.

        Returns
        -------
        tf.Tensor
            Quantized output tensor.
        """
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer.
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        """
        Calculates the indices of the closest embedding vectors.

        Parameters
        ----------
        flattened_inputs : tf.Tensor
            Flattened input tensor.

        Returns
        -------
        tf.Tensor
            Indices of the closest embedding vectors.
        """
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices


# VQVAE Encoder
def get_encoder(latent_dim=16):
    """
    Creates an encoder model for the VQ-VAE.

    Parameters
    ----------
    latent_dim : int, optional
        Dimensionality of the latent space, by default 16.

    Returns
    -------
    keras.Model
        Encoder model.
    """
    # Define the input layer
    encoder_inputs = keras.Input(shape=(image_size[0], image_size[1], 1))

    # Define the layers of the encoder
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same",
                      input_shape=image_size)(encoder_inputs)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)

    # Output layer representing the latent space
    encoder_outputs = layers.Conv2D(latent_dim, 1, padding="same")(x)

    return keras.Model(encoder_inputs, encoder_outputs, name="encoder")

# Decoder for the VQ-VAE
def get_decoder(latent_dim=16):
    """
    Creates a decoder model for the VQ-VAE.

    Parameters:
        latent_dim : int, optional
            Dimensionality of the latent space, by default 16.

    Returns:
        keras.Model
            Decoder model.
    """
    # Define the input layer representing the latent space
    latent_inputs = keras.Input(shape=get_encoder(latent_dim).output.shape[1:])

    # Define the layers of the decoder
    x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same")(
        latent_inputs)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)

    # Output layer representing the reconstructed image
    decoder_outputs = layers.Conv2DTranspose(1, 3, padding="same")(x)

    return keras.Model(latent_inputs, decoder_outputs, name="decoder")

# VQ-VAE model: inputs -> encoder -> quantizer -> decoder
def get_vqvae(latent_dim=16, num_embeddings=64):
    """
    Creates a VQ-VAE model.

    Parameters:
        latent_dim : int, optional
            Dimensionality of the latent space, by default 16.
        num_embeddings : int, optional
            Number of embedding vectors, by default 64.

    Returns
        keras.Model
            VQ-VAE model.
    """
    vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
    encoder = get_encoder(latent_dim)
    decoder = get_decoder(latent_dim)

    # Define the input layer for the entire model
    inputs = keras.Input(shape=(image_size[0], image_size[1], 1))

    # Define the flow of data through the model
    encoder_outputs = encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)
    reconstructions = decoder(quantized_latents)

    return keras.Model(inputs, reconstructions, name="vq_vae")

# Trainer for the VQ-VAE
class VQVAETrainer(keras.models.Model):
    """
    A class representing the trainer for the VQ-VAE model.

    ...

    Attributes
    ----------
    train_variance : float
        Training variance hyperparameter.
    latent_dim : int
        Dimensionality of the latent space.
    num_embeddings : int
        Number of embedding vectors.

    Methods
    -------
    train_step(x)
        Performs a training step for the VQ-VAE model.
    """

    def __init__(self, train_variance, latent_dim=32, num_embeddings=128, **kwargs):
        super(VQVAETrainer, self).__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        # Create a VQ-VAE model
        self.vqvae = get_vqvae(self.latent_dim, self.num_embeddings)

        # Trackers for different losses
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def train_step(self, x):
        """
        Performs a training step for the VQ-VAE model.

        Parameters:
            x : tf.Tensor
                Input tensor.

        Returns:
            dict
                Dictionary containing loss values.
        """
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }

# The code for this Pixel Convolution layer is
# https://keras.io/examples/generative/pixelcnn/
# Pixel Convolution Layer
class PixelConvLayer(layers.Layer):
    """
    A class representing the Pixel Convolution layer.

    ...

    Attributes
    ----------
    mask_type : str
        Type of mask applied to the layer.
    conv : keras.layers.Conv2D
        Convolutional layer.

    Methods
    -------
    build(input_shape)
        Builds the layer by initializing kernel variables.
    call(inputs)
        Performs the forward pass of the layer.
    """

    def __init__(self, mask_type, **kwargs):
        super(PixelConvLayer, self).__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)

    def build(self, input_shape):
        """
        Builds the layer by initializing kernel variables.

        Parameters:
            input_shape : tuple
                Shape of the input tensor.

        Returns:
            None
        """
        # Build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)
        # Use the initialized kernel to create the mask
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        """
        Performs the forward pass of the layer.

        Parameters:
            inputs : tf.Tensor
                Input tensor.

        Returns:
            tf.Tensor
                Output tensor.
        """
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)


# Residual Block
class ResidualBlock(keras.layers.Layer):
    """
    A class representing a Residual Block.

    ...

    Attributes
    ----------
    filters : int
        Number of filters in the convolutional layers.

    Methods
    -------
    call(inputs)
        Performs the forward pass of the layer.
    """

    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )
        self.pixel_conv = PixelConvLayer(
            mask_type="B",
            filters=filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same",
        )
        self.conv2 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )

    def call(self, inputs):
        """
        Performs the forward pass of the layer.

        Parameters:
            inputs : tf.Tensor
                Input tensor.

        Returns:
            tf.Tensor
                Output tensor.
        """
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return keras.layers.add([inputs, x])

# PixelCNN Model
def get_pixelcnn(
        num_residual_blocks,
        num_pixelcnn_layers,
        input_size=(8, 8),
        num_embeddings=128,
):
    """
    Creates a PixelCNN model.

    Parameters
    ----------
    num_residual_blocks : int
        Number of residual blocks in the model.
    num_pixelcnn_layers : int
        Number of PixelCNN layers in the model.
    input_size : tuple, optional
        Size of the input, by default (8, 8).
    num_embeddings : int, optional
        Number of embedding vectors, by default 128.

    Returns
    -------
    keras.Model
        PixelCNN model.
    """
    pixelcnn_inputs = keras.Input(shape=input_size, dtype=tf.int32)
    ohe = tf.one_hot(pixelcnn_inputs, num_embeddings)
    x = PixelConvLayer(
        mask_type="A", filters=128, kernel_size=7, activation="relu", padding="same"
    )(ohe)
    for _ in range(num_residual_blocks):
        x = ResidualBlock(filters=128)(x)

    for _ in range(num_pixelcnn_layers):
        x = PixelConvLayer(
            mask_type="B",
            filters=128,
            kernel_size=1,
            strides=1,
            activation="relu",
            padding="valid",
        )(x)
    out = keras.layers.Conv2D(
        filters=num_embeddings, kernel_size=1, strides=1, padding="valid"
    )(x)
    pixel_cnn = keras.Model(pixelcnn_inputs, out, name="pixel_cnn")

    return pixel_cnn