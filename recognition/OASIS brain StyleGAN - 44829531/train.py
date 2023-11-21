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
from util import SaveImage, SaveWeight
import matplotlib.pyplot as plt


class StyleGAN(keras.Model):
    """
    Implementation of the Style Generative Adversarial Network model using Keras
    """

    def __init__(self, epochs, batch_size):
        """
        Constructor for the StyleGAN class
        """
        super(StyleGAN, self).__init__()
        self.generator = Generator().generator()
        self.discriminator = Discriminator().discriminator()
        self.epochs = epochs
        self.batch_size = batch_size

    def compile(self):
        """
        Compile the StyleGAN model

        Sets up the optimisers, loss function and metrics
        """
        super(StyleGAN, self).compile()

        # initialise the optimisers
        self.generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.00001)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0000125)

        # initialise the loss function
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()

        # initialise the metrics
        self.generator_loss_metric = tf.keras.metrics.Mean(name="generator_loss")
        self.discriminator_loss_metric = tf.keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        """
        Return the metrics of the StyleGAN model
        """
        return [self.discriminator_loss_metric, self.generator_loss_metric]

    def plot_loss(self, epoch_history, filepath):
        """
        Visualise the loss values of the StyleGAN model by plotting the
        losses of the discriminator and generator against the number of epochs.
        """
        # Extract the discriminator and generator loss values
        # from the epoch_history object.
        discriminator_loss = epoch_history.epoch_history["discriminator_loss"]
        generator_loss = epoch_history.epoch_history["generator_loss"]

        # Plot the discriminator loss values against the number of epochs.
        minimum_value = min(min(generator_loss, discriminator_loss)) - 0.1
        maximum_value = max(max(generator_loss, discriminator_loss)) + 0.1

        # Plot the generator loss values against the number of epochs.
        plt.plot(discriminator_loss, label="Discriminator Loss")
        plt.plot(generator_loss, label="Generator Loss")

        # Set the title and labels of the plot
        plt.title("StyleGAN Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        # Set the y-axis limits to the minimum and maximum values
        plt.ylim([minimum_value, maximum_value])

        # If filepath is not None, save the plot to the filepath
        if filepath != "":
            directory_name = os.path.dirname(filepath)
            filepath = os.path.join(directory_name, filepath)
            plt.savefig("{}\loss_plot.png".format(filepath))

    def train(self, dataset_path, result_image_path, image_count,
              result_weight_path, plot_loss):

        """
        Train the StyleGAN model using the given dataset
        """
        # Create a list of callbacks to be used during training
        callbacks = []

        # Add the callback to save the model's weights
        if result_image_path != "":
            callbacks.append(SaveImage(result_image_path, image_count))
        if result_weight_path != "":
            callbacks.append(SaveWeight(result_weight_path))

        # Load the dataset
        images = load_data(dataset_path)

        # Compile the model
        self.compile()

        epoch_history = self.fit(images, epochs=self.epochs,
                                 callbacks=callbacks)

        if plot_loss:
            self.plot_loss(epoch_history, result_image_path)

    @tf.function
    def train_step(self, real_images):
        """
        Executes a single training step for both the generator
        and discriminator models.
        """
        # Train the generator and discriminator models
        generator_loss = self.train_generator()
        discriminator_loss = self.train_discriminator(real_images)

        # Update the metrics with the current loss values
        self.generator_loss_metric.update_state(generator_loss)
        self.discriminator_loss_metric.update_state(discriminator_loss)

        # Return the updated loss values for the generator and discriminator
        return {
            "generator_loss": self.generator_loss_metric.result(),
            "discriminator_loss": self.discriminator_loss_metric.result()
        }

    def train_generator(self):
        """
        Trains the generator model for one step.
        """
        # Get the inputs for the generator model.
        inputs = self.get_generator_inputs()

        # Record operations for automatic differentiation.
        with tf.GradientTape() as g_tape:
            # Generate fake images and get the discriminator's
            fake_images = self.generator(inputs)
            predictions = self.discriminator(fake_images)

            # Create labels indicating that the fake images are real.
            labels = tf.zeros([self.batch_size, 1])

            # Calculate the generator's loss.
            generator_loss = self.loss_fn(labels, predictions)

            # Get the trainable variables of the generator and calculate
            # the gradients of the generator's loss with respect to the
            # trainable variables
            trainable_variables = self.generator.trainable_variables
            gradients = g_tape.gradient(generator_loss, trainable_variables)

            # Apply the gradients to the generator's optimiser.
            self.generator_optimizer.apply_gradients(
                zip(gradients, trainable_variables))

        return generator_loss

    def train_discriminator(self, real_images):
        """
        Trains the discriminator model for one step using
        both real and generated images
        """
        # Generate fake images
        inputs = self.get_generator_inputs()
        generated_images = self.generator(inputs)

        # Create labels for the fake and real images and combine them.
        images = tf.concat([generated_images, real_images], axis=0)
        labels = tf.concat([tf.ones([self.batch_size, 1]),
                            tf.zeros([self.batch_size, 1])], axis=0)

        # Start recording operations for automatic differentiation.
        with tf.GradientTape() as d_tape:
            # Get the discriminator's predictions for the combined
            # batch of images and calculate the loss.
            predictions = self.discriminator(images)
            discriminator_loss = self.loss_fn(labels, predictions)

            # Get the trainable variables of the discriminator and calculate
            # the gradients of the discriminator loss with respect to them.
            gradients = d_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        return discriminator_loss

    def get_generator_inputs(self):
        """
        Generate the inputs for the generator model
        """
        # Generate latent space noise tensors.
        z = [tf.random.normal((self.batch_size, 512)) for i in range(7)]

        # Generate noise tensors for unique resolutions.
        noise = [tf.random.uniform((self.batch_size, res, res, 1)) for res in
                 [4, 8, 16, 32, 64, 128, 256]]

        # Create a constant input tensor.
        input = tf.ones([self.batch_size, 4, 4, 512])
        return [input, z, noise]


