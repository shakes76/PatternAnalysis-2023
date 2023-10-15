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

dataset_path = "C:\\Users\\ethan\\Desktop\\COMP3710" \
                     "\\keras_png_slices_train "
result_image_path = "figures"
result_weight_path = "figures"
result_image_count = 5

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

    def generator_inputs(self):
        z = [tf.random.normal((self.batch_size, 512)) for i in range(7)]
        noise = [tf.random.normal((self.batch_size, res, res, 1))
                 for res in [4, 8, 16, 32, 64, 128, 256]]
        input = tf.ones([self.batch_size, 4, 4,  512])
        return [input, z, noise]

    def train_generator(self):
        inputs = self.generator_inputs()
        with tf.GradientTape() as g_tape:
            fake_images = self.generator(inputs)
            predictions = self.discriminator(fake_images)
            labels = tf.zeros([self.batch_size, 1])
            generator_loss = self.loss(labels, predictions)
            trainable_variables = self.generator.trainable_variables
            gradients = g_tape.gradient(generator_loss, trainable_variables)
            self.generator_optimizer.apply_gradients(zip(gradients,
                                                         trainable_variables))
        return generator_loss

    def train_discriminator(self, real_images):
        inputs = self.generator_inputs()
        generated_images = self.generator(inputs)
        images = tf.concat([generated_images, real_images], axis=0)
        labels = tf.concat([tf.zeros([self.batch_size, 1]),
                            tf.ones([self.batch_size, 1])], axis=0)
        with tf.GradientTape() as d_tape:
            predictions = self.discriminator(images)
            discriminator_loss = self.loss(labels, predictions)
            trainable_variables = self.discriminator.trainable_variables
            gradients = d_tape.gradient(discriminator_loss, trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(gradients,
                                                             trainable_variables))
        return discriminator_loss

    def train(self, dataset_path, result_image_path, result_weight_path,
              result_image_count):
        dataset = load_data(dataset_path)
        dataset = dataset.shuffle(1000).batch(self.batch_size)

        self.setup()

        for epoch in range(self.epochs):
            print("Epoch: ", epoch)
            for real_images in dataset:
                losses = self.train_step(real_images)
                print("Generator loss: ", losses["generator_loss"])
                print("Discriminator loss: ", losses["discriminator_loss"])
            self.generate_and_save_images(epoch, result_image_path,
                                          result_image_count)
            self.save_weights(result_weight_path)

    def train_step(self, real_images):
        generator_loss = self.train_generator()
        discriminator_loss = self.train_discriminator(real_images)
        self.generator_loss_metric.update_state(generator_loss)
        self.discriminator_loss_metric.update_state(discriminator_loss)
        return {
            "generator_loss": self.generator_loss_metric.result(),
            "discriminator_loss": self.discriminator_loss_metric.result()
        }