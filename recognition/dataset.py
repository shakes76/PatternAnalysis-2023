import tensorflow as tf
import os

img_height = 240
img_width = 256

train = tf.keras.utils.image_dataset_from_directory(
    os.getcwd() + "/AD_NC/train",
    #"D:/COMP3710 Project/Siamese/recognition/AD_NC/train",
    labels="inferred",
    image_size=(img_height, img_width),
    batch_size=32)

test = tf.keras.utils.image_dataset_from_directory(
    os.getcwd() + "/AD_NC/test",
    #"D:/COMP3710 Project/Siamese/recognition/AD_NC/test",
    labels="inferred",
    image_size=(img_height, img_width),
    batch_size=32)

# Function to apply rgb_to_grayscale and random_flip_left_right
def preprocess_images(image, label):
    # Convert image to grayscale
    image = tf.image.rgb_to_grayscale(image)
    
    # Randomly flip the image horizontally
    image = tf.image.random_flip_left_right(image)
    
    # Normalize pixel values to be between 0 and 1
    image = tf.cast(image, tf.float32) / 255.0
    
    return image, label

# Apply the preprocessing function to the training dataset
train = train.map(preprocess_images)

def get_data():
    return (train, test)