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
    def __int__(self, num_enc, latent_dim, beta=0.25, layer_name="VQ"):
        super().__init__(layer_name=layer_name)
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
    def __int__(self, img_size=28, latent_dim=16, name='encoder'):
        """
        Defined Encoder for VQ-VAE
        :return: None
        """
        super(Encoder, self).__init__(name=name)
        self._latent_dim = latent_dim
        self._img_size = img_size

        self.input1 = layers.InputLayer(input_shape=self._img_size)
        self.conv1 = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')
        self.conv2 = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')
        self.conv3 = layers.Conv2D(self._latent_dim, 1, padding='same')


    def call(self, inputs):
        """
        Forward computation
        :param inputs: Inputs of the layer
        :return: Outputs of the layer
        """
        inputs = self.input1(inputs)
        hidden = self.conv1(inputs)
        hidden = self.conv2(hidden)
        return self.conv3(hidden)

class Decoder(Model):
    def __int__(self, input_dim=16, name='decoder'):
        """
        Defined decoder portion of VQ-VAE
        :return: None
        """
        super(Decoder, self).__init__(name=name)
        self.input1 = layers.InputLayer(input_shape=input_dim)
        self.convt_1 = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')
        self.convt_2 = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')
        self.convt_3 = layers.Conv2DTranspose(1, 3, padding='same')

    def call(self, inputs):
        """
        Forward computation for layer
        :param inputs: Inputs to the layer
        :return: Outputs of the layer
        """
        inputs = self.input1(inputs)
        hidden = self.convt_1(inputs)
        hidden = self.convt_2(hidden)
        return self.convt_3(hidden)

class VQVAE(Model):
    def __int__(self, tr_var, img_size=28, num_encoded=64, latent_dim=16, beta=0.25, name='vq_vae'):
        """
        Defines VQVAE
        :return: None
        """
        super(VQVAE, self).__init__(name=name)
        self._tr_var = tr_var
        self._img_size = img_size
        self._num_enc = num_encoded
        self._latent_dim = latent_dim
        self._beta = beta

        self._encoder = Encoder(self._img_size, self._latent_dim)
        self._vq = VQ(self._num_enc, self._latent_dim, self._beta)
        self._decoder = Decoder(self._encoder.output.shape[1:])

        self._total_loss = tf.keras.metrics.Mean(name='total_loss')
        self._vq_loss = tf.keras.metrics.Mean(name='vq_loss')
        self._reconstruction_loss = tf.keras.metrics.Mean(name='reconstruction_loss')


    def call(self, inputs):
        """
        Forward computation handler
        :param inputs: Inputs to the layer
        :return: Outputs of the layer
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
            recon = self.vqvae(data)

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

