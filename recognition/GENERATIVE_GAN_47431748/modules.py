"""
Contains the source code for the model
JACK CASHMAN - 47431748
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
import numpy as np

class VQ(layers.Layer):
    """
    Vector-Quantiser later of VQ-VAE
    """
    def __int__(self, num_enc, enc_dim, beta=0.25, layer_name="VectorQuant"):
        super().__init__(layer_name=layer_name)
        self._enc_dim = enc_dim
        self._num_enc = num_enc
        self._beta = beta

        runif_initialiser = tf.random_uniform_initializer()
        enc_shape = self._enc_dim, self._num_enc
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
                tf.reduce_sum(self._encoded ** 2, axis=0) -
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
        flat_inp = tf.reshape(inputs, [-1, self._enc_dim])

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
    def __int__(self):
        pass

    def call(self, inputs):
        pass

class Decoder(Model):
    def __int__(self):
        pass

    def call(self, inputs):
        pass

class VQVAE(Model):
    def __int__(self):
        pass

    def call(self, inputs):
        pass
