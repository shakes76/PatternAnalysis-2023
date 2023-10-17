# imports
import tensorflow as tf
import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array


# get dataset, using pre-processed ADNI dataset from blackboard



def create_data():
    trainDataDir = './ADNI_AD_NC_2D/AD_NC/train'
    testDataDir = './ADNI_AD_NC_2D/AD_NC/test'

    # training data paths
    AD_train_path = os.listdir(trainDataDir + "/AD")
    NC_train_path = os.listdir(trainDataDir + "/NC")
    #  testing data paths
    AD_test_path = os.listdir(testDataDir + "/AD")
    NC_test_path = os.listdir(testDataDir + "/NC")

    # image size is (240, 256, 3)
    # create training data
    training = [] # array containing train data

    for i in AD_train_path:
        image = load_img(trainDataDir + "/AD/" + i, target_size=(128, 128, 3))
        image = img_to_array(image)
        training.append([image, 1]) # append with 1 label

    for i in NC_train_path:
        image = load_img(trainDataDir + "/NC/" + i, target_size=(128, 128, 3))
        image = img_to_array(image)
        training.append([image, 0]) # append with 0 label

    # shuffle data to randomise?
    training_images = []
    training_labels = []
    for _ in training:
        training_images.append(_[0]) # image
        training_labels.append(_[1]) # label

    x_train = tf.convert_to_tensor(np.array(training_images, dtype=np.uint8))
    x_train = tf.cast(x_train, tf.float16) / 255.0
    y_train = tf.convert_to_tensor(training_labels)

    # create testing data
    testing = []  # array containing test data

    for i in AD_test_path:
        image = load_img(testDataDir + "/AD/" + i, target_size=(128, 128, 3))
        image = img_to_array(image)
        testing.append([image, 1])  # append with 1 label

    for i in NC_test_path:
        image = load_img(testDataDir + "/NC/" + i, target_size=(128, 128, 3))
        image = img_to_array(image)
        testing.append([image, 0])  # append with 0 label

    # shuffle data to randomise?
    testing_images = []
    testing_labels = []
    for _ in testing:
        testing_images.append(_[0])  # image
        testing_labels.append(_[1])  # label

    x_test = tf.convert_to_tensor(np.array(testing_images, dtype=np.uint8))
    x_test = tf.cast(x_test, tf.float16) / 255.0
    y_test = tf.convert_to_tensor(testing_labels)

    return x_train, y_train, x_test, y_test


