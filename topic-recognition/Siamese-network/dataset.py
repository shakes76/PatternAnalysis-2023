# dataset.py containing the data loader for loading and preprocessing your data

import os
import random
import keras as k
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.models import Model
from xmlrpc.client import Boolean
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split


# Paths to the dataset folder
AD_PATH = '/Users/jollylogan/TryTry/AD_NC/train/AD'
NC_PATH = '/Users/jollylogan/TryTry/AD_NC/train/NC'

AD_TEST_PATH = '/Users/jollylogan/TryTry/AD_NC/test/AD'
NC_TEST_PATH = '/Users/jollylogan/TryTry/AD_NC/test/NC'


def make_pairs(x, y):
    ### reference: https://keras.io/examples/vision/siamese_contrastive/
    """Creates a tuple containing image pairs with corresponding label.

    Arguments:
        x: List containing images, each index in this list corresponds to one image.
        y: List containing labels, each label with datatype of `int`.

    Returns:
        Tuple containing two numpy arrays as (pairs_of_samples, labels),
        where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
        labels are a binary array of shape (2len(x)).
    """

    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    pairs = []
    labels = []

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = random.choice(digit_indices[label1])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [0]

        # add a non-matching example
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [1]

    return np.array(pairs), np.array(labels).astype("float32")


def data_loader():
    ### reference: https://keras.io/examples/vision/siamese_contrastive/

    # Get all the paths to the images in the directories
    ad_path = [os.path.join(AD_PATH, path) for path in os.listdir(AD_PATH)]
    nc_path = [os.path.join(NC_PATH, path) for path in os.listdir(NC_PATH)]

    X_data = []
    X_data_labels = []

    # Load images in the AD train path
    for fpath in ad_path:
        image = Image.open(fpath)
        X_data.append(np.array(image))
        X_data_labels.append(1)
        image.close()

    # Load images in the NC train path
    for fpath in nc_path:
        image = Image.open(fpath)
        X_data.append(np.array(image))
        X_data_labels.append(0)
        image.close()
    
    # Convert to numpy array
    X_data = np.array(X_data)
    X_data_labels = np.array(X_data_labels)

    # Make train, validation and test sets
    x_train, x_test, y_train, y_test = train_test_split(X_data, X_data_labels, test_size=0.2, random_state=42, shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, shuffle=True)
    
    # Make train pairs
    pairs_train, labels_train = make_pairs(x_train, y_train)

    # Make validation pairs
    pairs_val, labels_val = make_pairs(x_val, y_val)

    # Make test pairs
    pairs_test, labels_test = make_pairs(x_test, y_test)

    # Split the training pairs
    x_train_1 = pairs_train[:, 0]
    x_train_2 = pairs_train[:, 1]

    # Split the validation pairs
    x_val_1 = pairs_val[:, 0]
    x_val_2 = pairs_val[:, 1]

    # Split the test pairs
    x_test_1 = pairs_test[:, 0]
    x_test_2 = pairs_test[:, 1]

    return (x_train_1, x_train_2), labels_train, (x_val_1, x_val_2), labels_val, (x_test_1, x_test_2), labels_test, X_data, X_data_labels


def data_test_loader():

    # Get all the paths to the images in the directories
    ad_test_path = [os.path.join(AD_TEST_PATH, path) for path in os.listdir(AD_TEST_PATH)]
    nc_test_path = [os.path.join(NC_TEST_PATH, path) for path in os.listdir(NC_TEST_PATH)]

    X_data = []
    X_data_labels = []

    # Load images in the AD test path
    for fpath in ad_test_path:
        image = Image.open(fpath)
        X_data.append(np.array(image))
        X_data_labels.append(0)
        image.close()

    # Load images in the NC test path
    for fpath in nc_test_path:
        image = Image.open(fpath)
        X_data.append(np.array(image))
        X_data_labels.append(0)
        image.close()
    
    # Convert to numpy array
    X_test_data = np.array(X_data)
    X_test_data_labels = np.array(X_data_labels)

    return X_test_data, X_test_data_labels