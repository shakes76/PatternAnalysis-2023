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

class FileSaver(keras.callbacks.Callback):
    def __init__(self):
        directory_name = os.path.dirname(__file__)
        self.result_image_path = os.path.join(directory_name, "figures")
        self.result_weight_path = os.path.join(directory_name, "figures")
        