# predict.py showing example usage of the trained model. 
# Print out any results 

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from dataset import data_test_loader


# Path where the model is saved
save_trained_classifier = '/Users/jollylogan/TryTry/Classifier.h5'

# Load the model
classifier_model = tf.keras.models.load_model(save_trained_classifier)

# Load the test data
X_test_data, X_test_data_labels = data_test_loader()

# Evaluate the model on the test data
classifier_model.evaluate(X_test_data, X_test_data_labels)

