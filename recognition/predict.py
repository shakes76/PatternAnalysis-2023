import numpy as np
from keras.layers import *
from keras.optimizers import Adam

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

model.compile(loss='binary_crossentropy', # As we are using Sigmoid activation
              optimizer=Adam(learning_rate=0.005),
              metrics=['accuracy'])

# Load trained weights
model.load_weights("./Checkpoint/cp_p100.ckpt")

# The code below allows Matplotlib to show the images
# import os    
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def check_same_class(input_A, input_B):
    is_same_class = model.predict([input_A.reshape(1, 75, 80), input_B.reshape(1, 75, 80)]).flatten() > 0.5
    return is_same_class[0]

correct = 0
incorrect = 0

for i in range(25):
    for j in range(40):
        input_A, input_B = X_test[i][j], X_test[i+1][j]
        label_A, label_B = y_test[i][j], y_test[i+1][j]

        guess = check_same_class(input_A, input_B)
        actual = (label_A == label_B)

        if guess == actual:
            correct += 1
        else:
            incorrect += 1

accuracy = correct / (correct + incorrect)

print("Correct Guesses:", correct, "Incorrect Guesses:", incorrect)
print("Accuracy:", accuracy)

# Code to show images, if required
#import matplotlib.pyplot as plt
#plt.figure(dpi=50)
#plt.imshow(input_A)
#plt.imshow(input_B)