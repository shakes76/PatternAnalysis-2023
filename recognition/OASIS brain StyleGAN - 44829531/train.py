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


class StyleGAN(keras.Model):
    def __init__(self):
        super(StyleGAN, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def setup(self):
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
        return [self.generator_loss_metric, self.discriminator_loss_metric]