import tensorflow as tf

import os
import math
import numpy as np
from pathlib import Path

from tensorflow import keras
from keras import layers
from keras.utils import load_img
from keras.utils import array_to_img
from keras.utils import img_to_array
from keras.preprocessing import image_dataset_from_directory

from IPython.display import display

current_directory = os.getcwd()
relative_train_path = 'recognition/Normal_Difficulty/AD_NC/train'
relative_test_path = 'recognition/Normal_Difficulty/AD_NC/test'

dir_train = Path(os.path.join(current_directory, relative_train_path))
dir_test = Path(os.path.join(current_directory, relative_test_path))
crop_size = 300
upscale_factor = 3
input_size = crop_size // upscale_factor
batch_size = 8

train_ds = image_dataset_from_directory(
    dir_train,
    batch_size=batch_size,
    image_size=(crop_size, crop_size),
    validation_split=0.2,
    subset="training",
    seed=1337,
    label_mode=None,
)

valid_ds = image_dataset_from_directory(
    dir_train,
    batch_size=batch_size,
    image_size=(crop_size, crop_size),
    validation_split=0.2,
    subset="validation",
    seed=1337,
    label_mode=None,
)

def scaling(input_image):
    input_image = input_image / 255.0
    return input_image


# Scale from (0, 255) to (0, 1)
train_ds = train_ds.map(scaling)
valid_ds = valid_ds.map(scaling)

