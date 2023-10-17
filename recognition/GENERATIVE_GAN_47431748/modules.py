"""
Contains the source code for the model
JACK CASHMAN - 47431748
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, metrics
import numpy as np


class VQ(layers.Layer):
    """
    Vector-Quantiser later of VQ-VAE
    """
    def __init__(self, num_enc, latent_dim, beta=0.25, name="VQ"):
        super(VQ, self).__init__(name=name)
        self._latent_dim = latent_dim
        self._num_enc = num_enc
        self._beta = beta

        runif_initialiser = tf.random_uniform_initializer()
        enc_shape = self._latent_dim, self._num_enc
        self._encoded = tf.Variable(initial_value=runif_initialiser(shape=enc_shape, dtype='float32'))

    def get_encoded(self):
        """
        Return the encoded vectors
        :return: The encoded vectors
        """
        return self._encoded

    def get_codebook_indices(self, inputs):
        """
        Return codebook indices of the encoding. Used in PCNN + VQVAE
        :param encoded: Encoded vectors
        :param inputs: Inputs, flattened
        :return:  Index of closest codebook vector
        """
        norms = (
                tf.reduce_sum(inputs ** 2, axis=1, keepdims=True) +
                tf.reduce_sum(self._encoded ** 2, axis=0, keepdims=True) -
                2 * tf.linalg.matmul(inputs, self._encoded)
        )
        return tf.argmin(norms, axis=1)

    def call(self, inputs):
        """
        Forward computation handler
        :param inputs: Inputs into layer
        :return: Outputs of layer
        """
        # Flatten input + Store dimensions
        og_shape = tf.shape(inputs)
        flat_inp = tf.reshape(inputs, [-1, self._latent_dim])

        # Quantise
        enc_idx = self.get_codebook_indices(flat_inp)
        onehot_idx = tf.one_hot(enc_idx, self._num_enc)
        qtised = tf.reshape(tf.linalg.matmul(onehot_idx, self._encoded, transpose_b=True), og_shape)

        # Calculates VQ loss from **insert ref**
        qtised_loss = tf.reduce_mean((qtised - tf.stop_gradient(inputs)) ** 2)
        commitment_loss = tf.reduce_mean((tf.stop_gradient(qtised) - inputs) ** 2)
        self.add_loss(self._beta * commitment_loss + qtised_loss)   # total loss in train loop

        return inputs + tf.stop_gradient(qtised - inputs)


class Encoder(Model):
    def __init__(self, latent_dim=16, name='encoder'):
        """
        Defined Encoder for VQ-VAE
        :return: None
        """
        super(Encoder, self).__init__(name=name)
        self._latent_dim = latent_dim

        self.conv1 = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')
        self.conv2 = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')
        self.conv3 = layers.Conv2D(self._latent_dim, 1, padding='same')

    def call(self, inputs):
        """
        Forward computation
        :param inputs: Inputs of the block
        :return: Outputs of the block
        """
        hidden = self.conv1(inputs)
        hidden = self.conv2(hidden)
        return self.conv3(hidden)


class Decoder(Model):
    def __int__(self, name='decoder', num_channels=1, **kwargs):
        """
        Defined decoder portion of VQ-VAE
        :return: None
        """
        super(Decoder, self).__init__(name=name, **kwargs)
        self._num_channels = num_channels
        self.conv_t1 = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')
        self.conv_t2 = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')
        self.conv_t3 = layers.Conv2DTranspose(1, 3, padding='same')

    def call(self, inputs):
        """
        Forward computation for block
        :param inputs: Inputs to the block
        :return: Outputs of the block
        """
        hidden = self.conv_t1(inputs)
        hidden = self.conv_t2(hidden)
        return self.conv_t3(hidden)


class VQVAE(Model):
    def __init__(self, tr_var, num_encoded=64, latent_dim=16, beta=0.25, num_channels=3, name='vq_vae', **kwargs):
        """
        Defines VQVAE
        :return: None
        """
        super(VQVAE, self).__init__(name=name)
        self._tr_var = tr_var
        self._num_enc = num_encoded
        self._num_channels = num_channels
        self._latent_dim = latent_dim
        self._beta = beta

        self._encoder = Encoder(self._latent_dim)
        self._vq = VQ(self._num_enc, self._latent_dim, self._beta)
        self._decoder = Decoder(num_channels=self._num_channels)

        self._total_loss = metrics.Mean(name='total_loss')
        self._vq_loss = metrics.Mean(name='vq_loss')
        self._reconstruction_loss = metrics.Mean(name='reconstruction_loss')

    def call(self, inputs):
        """
        Forward computation handler
        :param inputs: Inputs to the model
        :return: Outputs of the model
        """
        enc = self._encoder(inputs)
        quant = self._vq(enc)
        decoded = self._decoder(quant)
        return decoded

    def train_step(self, data):
        """
        Performs a single iteration of training and returns loss
        :param data: Input data
        :return: Loss vals
        """
        with tf.GradientTape() as tape:
            recon = self(data)

            recon_loss = (tf.reduce_mean((data - recon) ** 2) / self._tr_var)
            total_loss_val = recon_loss + sum(self.get_vq().losses)

        # Backprop
        grads = tape.gradient(total_loss_val, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update loss
        self._total_loss.update_state(total_loss_val)
        self._reconstruction_loss.update_state(recon_loss)
        self._vq_loss.update_state(sum(self.get_vq().losses))

        # Log results
        return {
            'loss': self._total_loss.result(),
            'vq_loss': self._vq_loss.result(),
            'reconstruction_loss': self._reconstruction_loss.result()
        }

    def test_step(self, data):
        """
        Performs single iteration of evaluation and returns losses
        :param data: Input data
        :return: Loss values
        """
        return self.train_step(data)

    @property
    def metrics(self):
        """
        Return metrics
        :return: Metrics
        """
        return [
            self._total_loss,
            self._vq_loss,
            self._reconstruction_loss
        ]

    def get_encoder(self):
        """
        Returns encoder
        :return: Encoder of the VQVAE
        """
        return self._encoder

    def get_vq(self):
        """
        Returns VQ Layer
        :return: VQ Layer
        """
        return self._vq

    def get_decoder(self):
        """
        Returns decoder
        :return: Decoder
        """
        return self._decoder


"""
Pixel CNN Implementation
"""


class PixelConv(layers.Layer):
    """
    Custom convolutions layer + masking
    """
    def __init__(self, kernel_mask_type, name='pixel_conv', **kwargs):
        super(PixelConv, self).__init__(name=name)
        self._main_conv = layers.Conv2D(**kwargs)

        self._kernel_mask_type = kernel_mask_type

    def build(self, input_shape):
        """
        Define layer variables
        :param input_shape: Input layer shape
        :return: None
        """
        self._main_conv.build(input_shape)
        kernel_shape = self._main_conv.kernel.get_shape()
        self._kernel_mask = np.zeros(shape=kernel_shape)
        self._kernel_mask[:kernel_shape[0] // 2, ...] = 1.0
        self._kernel_mask[kernel_shape[0] // 2, :kernel_shape[1] // 2, ...] = 1.0
        if self._kernel_mask_type == "B":   # Include middle pixel
            self._kernel_mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        """
        Forward computation handler of this layer
        :param inputs: Inputs to the layer
        :return: Outputs of the layer
        """
        self._main_conv.kernel.assign(self._main_conv.kernel * self._kernel_mask)
        return self._main_conv(inputs)


class PixelResidualBlock(layers.Layer):
    def __init__(self, num_filters, name='pixel_res_block'):
        super(PixelResidualBlock, self).__init__(name=name)
        self._num_filters = num_filters
        self._conv1 = layers.Conv2D(filters=self._num_filters, kernel_size=1, activation="relu",
                                    padding="same")
        self._pixel_conv = PixelConv(filters=self._num_filters // 2,
                                     kernel_size=3, activation="relu",
                                     padding="same", kernel_mask_type="B")
        self._conv2 = layers.Conv2D(filters=self._num_filters, kernel_size=1, activation="relu",
                                    padding="same")

    def call(self, inputs):
        """
        Forward computation handler of this layer
        :param inputs: Inputs to the layer
        :return: Outputs of the layer
        """
        hidden = self._conv1(inputs)
        hidden = self._pixel_conv(hidden)
        hidden = self._conv2(hidden)

        return inputs + hidden


class PixelCNN(Model):
    """
    PixelCNN generative model
    """
    def __init__(self, num_res=2, num_pixel_B=2, num_encoded=128, num_filters=128,
                 kernel_size=7, activation='relu', name='pixel_cnn'):
        super(PixelCNN, self).__init__(name=name)
        self._num_res = num_res
        self._num_pixel_B = num_pixel_B
        self._num_encoded = num_encoded
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._activation = activation
        self._total_loss = metrics.Mean(name="total_loss")

        self._pixel_A = PixelConv(kernel_mask_type="A", filters=self._num_filters,
                                  kernel_size=self._kernel_size, activation=self._activation)
        self._pixel_resids = [PixelResidualBlock(num_filters=self._num_filters)
                              for _ in range(self._num_res)]
        self._pixel_Bs = [PixelConv(kernel_mask_type="B", filters=self._num_filters, kernel_size=1,
                                    activation=self._activation)
                          for _ in range(self._num_pixel_B)]
        self._conv = layers.Conv2D(filters=self._num_encoded, kernel_size=1)

    def call(self, inputs):
        """
        Forward computation of this model
        :param inputs: Inputs to the model
        :return: Outputs of the model
        """
        inputs = tf.cast(inputs, dtype=tf.int32)
        inputs = tf.one_hot(inputs, self._num_encoded)
        hidden = self._pixel_A(inputs)
        for i in range(self._num_res):
            hidden = self._pixel_resids[i](hidden)
        for j in range(self._num_pixel_B):
            hidden = self._pixel_Bs[j](hidden)
        return self._conv(hidden)

    def train_step(self, data):
        """
        Performs single iter of training
        :param data: Input data
        :return: Loss values
        """
        with tf.GradientTape() as tape:
            loss_val = self.compiled_loss(data, self(data))

        # back prop
        grads = tape.gradient(loss_val, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self._total_loss.update_state(loss_val)

        return {
            'loss': self._total_loss.result()
        }

    def test_step(self, data):
        """
        Performs one iteration of evaluation and returns loss values
        :param data: Input data
        :return: Loss values
        """
        return self.train_step(data)
