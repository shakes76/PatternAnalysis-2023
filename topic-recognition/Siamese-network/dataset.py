import os
import random
import keras as k
import numpy as np
import tensorflow as tf
import keras.layers as kl
import keras.backend as kb
from keras.models import Model
from xmlrpc.client import Boolean
from tensorflow.keras.models import load_model
from matplotlib import image
from matplotlib import pyplot
import matplotlib.pyplot as plt


AD_PATH = '/Users/jollylogan/TryTry/AD_NC/train/AD'
NC_PATH = '/Users/jollylogan/TryTry/AD_NC/train/NC'

AD_TEST_PATH = '/Users/jollylogan/TryTry/AD_NC/test/AD'
NC_TEST_PATH = '/Users/jollylogan/TryTry/AD_NC/test/NC'


def load_train_data(batch_size=32):
    # Get all the paths to the images in the directories
    ad_paths = [os.path.join(AD_PATH, path) for path in os.listdir(AD_PATH)]
    nc_paths = [os.path.join(NC_PATH, path) for path in os.listdir(NC_PATH)]

    # Create tf.data.Dataset objects
    ad_ds = tf.data.Dataset.from_tensor_slices(ad_paths)
    nc_ds = tf.data.Dataset.from_tensor_slices(nc_paths)

    # Create pairs 
    pos_pair1 = tf.data.Dataset.zip((ad_ds, ad_ds)) # Positive pair (both from AD) 
    pos_pair2 = tf.data.Dataset.zip((nc_ds, nc_ds)) # Positive pair (both from NC) 
    neg_pair1 = tf.data.Dataset.zip((ad_ds, nc_ds)) # Negative pair (one from AD and one from AC) 
    neg_pair2 = tf.data.Dataset.zip((nc_ds, ad_ds)) # Negative pair (one from NC and one from AD) 
                                                    # - the same as previous combination 
                                                    # - the purpose is to make the amount of +ve pairs and -ve pairs balance

    num_pairs = min(len(ad_paths), len(nc_paths))
    # Limit the length of pos_pair2 to num_pairs
    pos_pair2 = pos_pair2.take(num_pairs)

    subset_size = 2600  
    pos_pair1 = pos_pair1.take(subset_size)
    pos_pair2 = pos_pair2.take(subset_size)
    neg_pair1 = neg_pair1.take(subset_size)
    neg_pair2 = neg_pair2.take(subset_size)

    # Shuffling tf.data.Dataset
    pos_pair1 = pos_pair1.shuffle(buffer_size=10000)  
    pos_pair2 = pos_pair2.shuffle(buffer_size=10000)
    neg_pair1 = neg_pair1.shuffle(buffer_size=10000)
    neg_pair2 = neg_pair2.shuffle(buffer_size=10000)
    print(len(pos_pair1))
    print(len(pos_pair2))
    print(len(neg_pair1))
    print(len(neg_pair2))

    # Concatenate the pairs
    pair_compare = pos_pair1.concatenate(pos_pair2).concatenate(neg_pair1).concatenate(neg_pair2)
    print(len(pair_compare))

    num_pairs = len(pair_compare)
    labels = np.concatenate([np.zeros([num_pairs//2]), np.ones([num_pairs//2])])
    labels = np.expand_dims(labels, -1)

    first_input = pair_compare.map(lambda x, y: tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(x), 1), [128, 128]) / 255)
    second_input = pair_compare.map(lambda x, y: tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(y), 1), [128, 128]) / 255)
    print(len(first_input))
    print(len(second_input))

    label_input = tf.data.Dataset.from_tensor_slices(labels)
    print(len(label_input))

    # Create a dataset where each element is ([base_image, pair_image], label)
    dataset = tf.data.Dataset.zip(((first_input, second_input), label_input)).shuffle(num_pairs)
    print(len(dataset))

    # Determine the number of images to use for training (80%)
    train_num = int(round(0.8 * num_pairs, 1))
    # Create the training and validation datasets
    train = dataset.take(train_num).batch(batch_size)
    val = dataset.skip(train_num).batch(batch_size)

    return train, val


def load_classify_data(batch_size=32):

    # Get all the paths to the images in the directories
    ad_path = [os.path.join(AD_PATH, path) for path in os.listdir(AD_PATH)]
    nc_path = [os.path.join(NC_PATH, path) for path in os.listdir(NC_PATH)]
    print(len(ad_path))

    num_ad = min(len(ad_path), len(nc_path)) 
    # Limit the amount of NC images to num_pairs (make it balance)
    nc_path = nc_path[:num_ad]
    print(len(nc_path))

    # Combine all images together
    paths = ad_path + nc_path
    print(len(paths))

    # Create labels for the images: 0 for AD and 1 for CN
    labels = np.concatenate([np.ones([len(ad_path)]), np.zeros([len(nc_path)])])
    labels = np.expand_dims(labels, -1)

    all_images = tf.data.Dataset.from_tensor_slices(paths)
    all_images = all_images.map(lambda x: tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(x), 1), [128, 128]) / 255)
    print(len(all_images))
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)

    # Create a  dataset from all the images with labels
    dataset = tf.data.Dataset.zip((all_images, labels_ds)).shuffle(len(paths))

    # Determine the number of images to use for training (80%)
    train_num = int(round(0.8 * len(dataset), 1))
    # Create the training and validation datasets
    train = dataset.take(train_num).batch(batch_size)
    val = dataset.skip(train_num).batch(batch_size)

    return train, val


def load_testing_data(batch_size=32):

    # Get all the paths to the images in the directories
    ad_path = [os.path.join(AD_TEST_PATH, path) for path in os.listdir(AD_TEST_PATH)]
    nc_path = [os.path.join(NC_TEST_PATH, path) for path in os.listdir(NC_TEST_PATH)]
    print(len(ad_path))

    num_ad = min(len(ad_path), len(nc_path)) 
    # Limit the length of pos_pair2 to num_pairs  
    nc_path = nc_path[:num_ad]
    print(len(nc_path))

    # Combine all images together
    paths = ad_path + nc_path
    print(len(paths))

    # Create labels for the images: 0 for AD and 1 for CN
    labels = np.concatenate([np.ones([len(ad_path)]), np.zeros([len(nc_path)])])
    labels = np.expand_dims(labels, -1)

    all_images = tf.data.Dataset.from_tensor_slices(paths)
    all_images = all_images.map(lambda x: tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(x), 1), [128, 128]) / 255)
    print(len(all_images))
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((all_images, labels_ds)).shuffle(len(paths))

    return dataset.batch(32)
