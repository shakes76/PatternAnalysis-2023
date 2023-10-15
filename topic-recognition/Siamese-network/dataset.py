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
CN_PATH = '/Users/jollylogan/TryTry/AD_NC/train/NC'

AD_TEST_PATH = '/Users/jollylogan/TryTry/AD_NC/test/AD'
CN_TEST_PATH = '/Users/jollylogan/TryTry/AD_NC/test/NC'


def load_train_data(batch_size=32):
    # Get all the paths to the images in the directories
    ad_paths = [os.path.join(AD_PATH, path) for path in os.listdir(AD_PATH)]
    cn_paths = [os.path.join(CN_PATH, path) for path in os.listdir(CN_PATH)]

    # Create tf.data.Dataset objects
    ad_ds = tf.data.Dataset.from_tensor_slices(ad_paths)
    cn_ds = tf.data.Dataset.from_tensor_slices(cn_paths)

    # Create pairs
    pos_pair1 = tf.data.Dataset.zip((ad_ds, ad_ds)) # Same images (both from AD)
    pos_pair2 = tf.data.Dataset.zip((cn_ds, cn_ds)) # Same images (both from NC) 
    neg_pair1 = tf.data.Dataset.zip((ad_ds, cn_ds)) # Different images (one from AD and one from AC)
    neg_pair2 = tf.data.Dataset.zip((cn_ds, ad_ds)) # Different images (one from NC and one from AD)

    num_pairs = min(len(ad_paths), len(cn_paths))
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






