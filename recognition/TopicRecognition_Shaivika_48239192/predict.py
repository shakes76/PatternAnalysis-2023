import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from modules import create_classifier, Patches  # Import the Patches layer
from dataset import load_and_preprocess_data
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from train import model

# Define the patch size
patch_size = 6

# Load an image and create the resized image
# image_size = 128  
# image = load_img('path_to_image.jpg', target_size=(image_size, image_size))
# image = img_to_array(image)
# resized_image = tf.convert_to_tensor([image])
# Load and preprocess data

Images, Labels = load_and_preprocess_data()

# Load the trained model
model = tf.keras.models.load_model('model/trained_model.h5')

# Make predictions
y_pred = model.predict(Images)
y_pred = np.argmax(y_pred, axis=1)

# Generate a confusion matrix
y_true = np.argmax(Labels, axis=1)
cm = confusion_matrix(y_true, y_pred)

# Plot a heatmap of the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Load and plot the training history
history = model.history
plt.figure(figsize=(15, 15))
plt.subplot(221)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.subplot(222)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()

best_accuracy = round(max(history.history["accuracy"]) * 100)
print(f'The model has a best accuracy of {best_accuracy}%')
