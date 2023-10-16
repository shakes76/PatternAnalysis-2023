"""
util.py

Author: Ethan Jones
Student ID: 44829531
COMP3710 OASIS brain StyleGAN project
Semester 2, 2023
"""

import os
import tensorflow as tf
from tensorflow import keras
import numpy as np


class SaveImage(keras.callbacks.Callback):

    def __init__(self, path, image_count=5):
        self.image_count = image_count
        directory_name = os.path.dirname(__file__)
        self.path = os.path.join(directory_name, path)

    def on_epoch_end(self, epoch, logs):
        generator_inputs = self.model.get_generator_inputs()
        output_images = self.model.generator(generator_inputs)
        output_images *= 256
        output_images.numpy()
        for i in range(self.image_count):
            img = keras.preprocessing.image.array_to_img(output_images[i])
            img.save("{}\epoch_{}_image_{}.png".format(self.path, epoch, i))


class SaveWeight(keras.callbacks.Callback):

    def __init__(self, path):
        directory_name = os.path.dirname(__file__)
        self.path = os.path.join(directory_name, path)

    def on_epoch_end(self, epoch, logs):
        self.model.generator.save_weights("{}\epoch_{}_generator.h5".format(self.path, epoch))
        self.model.discriminator.save_weights("{}\epoch_{}_discriminator.h5".format(self.path, epoch))
