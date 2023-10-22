import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from itertools import islice
import math

# Constants related to data preprocessing
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
CHANNELS = 3
SEED = 45
BATCH_SIZE = 2
# Directory containing the dataset
PATH_ORIGINAL_DATA = os.path.join("datasets", "training_input")  # directory that contains folder containing input images
PATH_SEG_DATA = os.path.join("datasets", "training_groundtruth")  # directory that contains folder containing ground truth images
IMAGE_MODE = "rgb" 
MASK_MODE = "grayscale"
DATA_TRAIN_GEN_ARGS = dict(
    rescale=1.0/255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=0.2)  # 0.2 used to have training set take the first 80% of images
# Set the properties for the image generators for testing images. No image transformations.
DATA_TEST_GEN_ARGS = dict(
    rescale=1.0/255,
    validation_split=0.8)  # 0.8 used to have test set take the final 20% of images (keep train/test data separated)
# Set the shared properties for generator flows - scale all images to given dimensions.
TEST_TRAIN_GEN_ARGS = dict(
    seed=SEED,
    class_mode=None,
    batch_size=BATCH_SIZE,
    interpolation="nearest",
    subset='training',  # all subsets are set to training - this corresponds to the first 80% and last 20% for each
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))

# Preprocess data forming the generators.
def pre_process_data():
    train_image_data_generator = keras.preprocessing.image.ImageDataGenerator(**DATA_TRAIN_GEN_ARGS)
    train_mask_data_generator = keras.preprocessing.image.ImageDataGenerator(**DATA_TRAIN_GEN_ARGS)
    test_image_data_generator = keras.preprocessing.image.ImageDataGenerator(**DATA_TEST_GEN_ARGS)
    test_mask_data_generator = keras.preprocessing.image.ImageDataGenerator(**DATA_TEST_GEN_ARGS)

    image_train_gen = train_image_data_generator.flow_from_directory(
        PATH_ORIGINAL_DATA,
        color_mode=IMAGE_MODE,
        **TEST_TRAIN_GEN_ARGS)

    image_test_gen = test_image_data_generator.flow_from_directory(
        PATH_ORIGINAL_DATA,
        color_mode=IMAGE_MODE,
        **TEST_TRAIN_GEN_ARGS)

    mask_train_gen = train_mask_data_generator.flow_from_directory(
        PATH_SEG_DATA,
        color_mode=MASK_MODE,
        **TEST_TRAIN_GEN_ARGS)

    mask_test_gen = test_mask_data_generator.flow_from_directory(
        PATH_SEG_DATA,
        color_mode=MASK_MODE,
        **TEST_TRAIN_GEN_ARGS)

    # Ideally this would be a Sequence joining the two generators instead of zipping them together to keep everything
    # thread-safe, allowing for multiprocessing - but if it ain't broke. (It works).
    return zip(image_train_gen, mask_train_gen), zip(image_test_gen, mask_test_gen)
