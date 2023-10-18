import tensorflow as tf
import os

img_height = 240
img_width = 256

train = tf.keras.utils.image_dataset_from_directory(
    os.getcwd() + "/AD_NC/train",
    #"D:/COMP3710 Project/Siamese/recognition/AD_NC/train",
    labels="inferred",
    image_size=(img_height, img_width),
    batch_size=None)

test = tf.keras.utils.image_dataset_from_directory(
    os.getcwd() + "/AD_NC/test",
    #"D:/COMP3710 Project/Siamese/recognition/AD_NC/test",
    labels="inferred",
    image_size=(img_height, img_width),
    batch_size=None)

def get_data():
    return (train, test)