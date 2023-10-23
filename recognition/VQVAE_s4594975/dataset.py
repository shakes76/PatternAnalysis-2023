import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import tensorflow as tf

# Initialise paths for training, testing and validation sets 
TRAIN_PATH = "/Users/pc/Documents/COMP3710/keras_png_slices_data/keras_png_slices_train"
TEST_PATH = "/Users/pc/Documents/COMP3710/keras_png_slices_data/keras_png_slices_test"
VAL_PATH = "/Users/pc/Documents/COMP3710/keras_png_slices_data/keras_png_slices_validate"
# Incitialise directories for training, testing and validation data
train_files = os.listdir(TRAIN_PATH)
test_files = os.listdir(TEST_PATH)
val_files = os.listdir(VAL_PATH)
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
train_images = load_images(TRAIN_PATH, train_files)
test_images = load_images(TEST_PATH, test_files)

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

#visualize data
def show(list):
    """
    Displays the given list of images.
    """
    plt.figure(figsize=(6, 6))

    title = ['Train', 'Test']

    for i in range(len(list)):
        plt.subplot(1, len(list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(list[i]), cmap=plt.cm.gray)
        plt.axis('off')
    
    plt.show()
# Print the item in the list
#show([x_train_scaled[0], x_test_scaled[0]])

