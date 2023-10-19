import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator

img_height = 240
img_width = 256

train = tf.keras.utils.image_dataset_from_directory(
    os.getcwd() + "/AD_NC/train",
    #"D:/COMP3710 Project/Siamese/recognition/AD_NC/train",
    labels="inferred",
    image_size=(img_height, img_width),
    batch_size=40)

test = tf.keras.utils.image_dataset_from_directory(
    os.getcwd() + "/AD_NC/test",
    #"D:/COMP3710 Project/Siamese/recognition/AD_NC/test",
    labels="inferred",
    image_size=(img_height, img_width),
    batch_size=40)


# Function to apply rgb_to_grayscale and random_flip_left_right
def preprocess_training_images(image, label):
    # Convert image to grayscale
    image = tf.image.rgb_to_grayscale(image)
    
    # Make the images smaller
    image = tf.image.resize(image, (75,80), preserve_aspect_ratio=True)
    
    # Randomly flip the image horizontally
    image = tf.image.random_flip_left_right(image)
    
    # Normalize pixel values to be between 0 and 1
    image = tf.cast(image, tf.float32) / 255.0
    
    return image, label

def preprocess_test_images(image, label):
    # Convert image to grayscale
    image = tf.image.rgb_to_grayscale(image)

    # Make the images smaller
    image = tf.image.resize(image, (75,80), preserve_aspect_ratio=True)
    
    # Normalize pixel values to be between 0 and 1
    image = tf.cast(image, tf.float32) / 255.0
    
    return image, label

# Apply the preprocessing function to the training dataset
train = train.map(preprocess_training_images)
test = test.map(preprocess_test_images)

def get_data():
    return (train, test)