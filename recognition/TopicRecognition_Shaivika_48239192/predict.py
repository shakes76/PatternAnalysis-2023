import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from modules import create_classifier
from dataset import load_and_preprocess_data
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load and preprocess data
Images, Labels = load_and_preprocess_data()

# Load the trained model
model = tf.keras.models.load_model('model/trained_model.h5')

# Make predictions
y_pred = model.predict(Images)
y_pred = np.argmax(y_pred, axis=1)