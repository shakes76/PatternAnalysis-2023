"""
train.py

Author: Ethan Jones
Student ID: 44829531
COMP3710 OASIS brain StyleGAN project
Semester 2, 2023
"""

import os
import tensorflow as tf
from tensorflow import keras
from dataset import load_data
from modules import Generator, Discriminator

LEARNING_RATE = 0.0001
EPOCHS = 80
BATCH_SIZE = 32


class StyleGAN(keras.Model):
    """
    Implementation of the Style Generative Adversarial Network model using Keras
    """
    def __init__(self, epochs, batch_size):
        """
        Constructor for the StyleGAN class
        """
        super(StyleGAN, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.epochs = epochs
        self.batch_size = batch_size

    def setup(self):
        """
        Setup the StyleGAN model

        Sets up the optimisers, loss function and metrics
        """
        super(StyleGAN, self).setup()

        # initialise the optimisers
        self.generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

        # initialise the loss function
        self.loss = tf.keras.losses.BinaryCrossentropy()

        # initialise the metrics
        self.generator_loss_metric = \
            tf.keras.metrics.Mean(name="generator_loss")
        self.discriminator_loss_metric = \
            tf.keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        """
        Return the metrics of the StyleGAN model
        """
        return [self.generator_loss_metric, self.discriminator_loss_metric]