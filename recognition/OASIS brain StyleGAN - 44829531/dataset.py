"""
dataset.py

Author: Ethan Jones
Student ID: 44829531
COMP3710 OASIS brain StyleGAN project
Semester 2, 2023
"""

import os
import tensorflow
from tensorflow import keras


def load_data(dataset_path="/home/groups/comp3710/OASIS/keras_png_slices_train"):
    """
    Load the dataset from the given path.

    param: dataset_path: The path to the dataset
    return: The dataset of images
    """
    directory_name = os.path.dirname(__file__)
    file_path = os.path.join(directory_name, dataset_path)

    # Scale image data from original to [0, 1]
    image_data = keras.preprocessing.image_dataset_from_directory(file_path, label_mode=None, color_mode="grayscale")
    image_data = image_data.map(lambda x: x / 255.0)

    return image_data

