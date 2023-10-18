import tensorflow as tf
import numpy as np
import keras.api._v2.keras as keras # Required as, though it violates Python conventions, my TF cannot load Keras properly
from keras.layers import *
from keras.models import Sequential, Model

import modules
import dataset

from itertools import product

train, test = dataset.get_data()
model = modules.get_model()

# Extract X and y from the train dataset
X_train = [X for X, _ in train]
y_train = [y for _, y in train]

# Do the same in test
X_test = [X for X, _ in test]
y_test = [y for _, y in test]

def create_pairs(X, y):
    X_pairs, ys = [], []
    data = [(x1, y1) for x1, y1 in zip(X, y)]

    for d in product(data, data): # Creates a Cartesian product of the list of tuples with itself
        x_A, y_A = d[0]
        x_B, y_B = d[1]

        # Check whether these two are the same class
        new_y = int(y_A == y_B)

        X_pairs.append([x_A, x_B])
        ys.append(new_y)
    
    X_pairs = np.array(X_pairs)
    ys = np.array(ys)

    return X_pairs, ys