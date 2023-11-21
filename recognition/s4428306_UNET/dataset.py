#Extensive use of tensorflow and keras documentation was used to write this code.

"""
This file is for loading and preprocessing the data.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from PIL import Image

#Taken from code I wrote for prac2.
def loadDataFrom(directory, channels, size=128):
    """
    Loads the images from the given directory as a tensorflow Dataset.
    """
    numberOfImages = len(os.listdir(directory))
    if channels > 1:
        data = np.zeros((numberOfImages, size, size, channels))
    else:
        data = np.zeros((numberOfImages, size, size))
    data_shape = (numberOfImages, size, size, channels)
    for i, imageName in enumerate(sorted(os.listdir(directory))):
        #Make sure not to load in the license file.
        if imageName != "LICENSE.txt" and imageName != "ATTRIBUTION.txt":
            imagePath = os.path.join(directory, imageName)
            if channels > 1:
                data[i, :, :, :] = np.asarray(Image.open(imagePath).resize((size, size)))
            else:
                data[i, :, :] = np.asarray(Image.open(imagePath).resize((size, size)))
    #Need the extra dimension for image manipulation.
    if channels == 1:
        data = np.reshape(data, data_shape)
    return tf.data.Dataset.from_tensor_slices(data)

#The following is modified code from:
#https://towardsdatascience.com/how-to-split-a-tensorflow-dataset-into-train-validation-and-test-sets-526c8dd29438
def partition(data, train_size, val_size, test_size, seed):
    """
    Partitions the data into training, test and validation sets.
    """
    data.shuffle(2596, seed)
    train_data = data.take(train_size)
    val_data = data.skip(train_size).take(val_size)
    test_data = data.skip(train_size).skip(val_size)
    return train_data, val_data, test_data

#Taken from code I wrote for prac2.
def normalize(image, mask):
    """
    Normalizes image values to be between 0 and 1.
    Converts masks to 0s and 1s.
    """
    image = tf.cast(image, tf.float64) / 255.0
    mask = tf.clip_by_value(mask, clip_value_min=0, clip_value_max=1)
    return image, mask

def preprocessing(batch_size=64):
    """
    Preprocesses the data so that it is ready for training and predictions.
    """
    #These are the directories for the datasets.
    test_dir = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Test_Input/"
    training_images_dir = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2/"
    training_gt_dir = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2/"
    #Load in and normalize data.
    image_data = loadDataFrom(training_images_dir, channels=3)
    mask_data = loadDataFrom(training_gt_dir, channels=1)
    isic_data = tf.data.Dataset.zip((image_data, mask_data))
    isic_data = isic_data.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #Split.
    train_data, val_data, test_data = partition(isic_data, 1796, 400, 400, seed=271828)
    #Create batches.
    train_batches = train_data.cache().batch(batch_size).repeat()
    train_batches = train_batches.prefetch(tf.data.experimental.AUTOTUNE)
    val_batches = val_data.batch(batch_size)
    test_batches = test_data.batch(batch_size)
    return train_batches, val_batches, test_batches

