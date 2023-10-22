import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import tensorflow as tf

# Load keras data set to colab 
from google.colab import drive
drive.mount('/content/drive')

!unzip /content/drive/My\ Drive/keras_png_slices_data.zip
# Initialise paths for training, testing and validation sets
TRAIN_IMAGE_PATH = "keras_png_slices_data/keras_png_slices_train"
TEST_IMAGE_PATH = "keras_png_slices_data/keras_png_slices_test"
VAL_IMAGE_PATH = "keras_png_slices_data/keras_png_slices_validate"
# Incitialise directories for training, testing and validation data
train_files = os.listdir(TRAIN_IMAGE_PATH)
test_files = os.listdir(TEST_IMAGE_PATH)
validate_files = os.listdir(VAL_IMAGE_PATH)
# Initialise image dimensions for model
IMG_HEIGHT = 80
IMG_WIDTH = 80

def load_images(path, image_files):
    """
    Returns a list of resized images at the given path.
    """
    images = []

    for file in image_files:
        image = Image.open(path + '/' + file) 
        image = image.resize((IMG_HEIGHT, IMG_WIDTH))
        image = np.reshape(image, (IMG_HEIGHT, IMG_WIDTH, 1))
        images.append(image)
        
    return images

# Load images into lists
train_images = load_images(TRAIN_IMAGE_PATH, train_files)
test_images = load_images(TEST_IMAGE_PATH, test_files)

# Convert lists into np arrays
x_train = np.array(train_images)
x_test = np.array(test_images)
# Normalise the OASIS brain data to be [-0.5, 0.5].
x_train_scaled = (x_train / 255.0) - 0.5
x_test_scaled = (x_test / 255.0) - 0.5

# Calculate data variance to normalise Mean Squared Error
data_variance = np.var(x_train / 255.0)

# Check shapes of arrays
print(x_train.shape)
print(data_variance.shape)

print(x_test.shape)