import numpy as np
from keras.layers import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import modules
import dataset

from itertools import product

train, test = dataset.get_data()
model = modules.get_model()

# Extract X and y from the train dataset
X_train = [X for X, _ in train]
y_train = [y for _, y in train]
X_train = np.array(X_train)
y_train = np.array(y_train)

# Do the same in test
X_test = [X for X, _ in test]
y_test = [y for _, y in test]
X_test = np.array(X_test)
y_test = np.array(y_test)

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

# Specify checkpoint paths
checkpoint_path = "./Checkpoint/cp_p100.ckpt"

# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                              save_weights_only=True,
                              verbose=1)

# Load existing checkpoint
model.load_weights("./Saved/cp.ckpt")

model.compile(loss='binary_crossentropy', # As we are using Sigmoid activation
              optimizer=Adam(learning_rate=0.005),
              metrics=['accuracy'])

for b in range(50):
    X_train_pairs, ys_train = create_pairs(X_train[b], y_train[b])
    X_test_pairs, ys_test = create_pairs(X_test[b], y_test[b])

    model.fit(x=[X_train_pairs[:, 0, :, :], X_train_pairs[:, 1, :, :]],
          y=ys_train,
          validation_data=([X_test_pairs[:, 0, :, :], X_test_pairs[:, 1, :, :]], ys_test),
          batch_size=None,
          steps_per_epoch=20,
          callbacks=[cp_callback])
