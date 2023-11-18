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

Images, Labels = load_and_preprocess_data()

# Load the trained model
train_indices, test_indices = train_test_split(list(range(Images.shape[0])), train_size=0.8, test_size=0.2, shuffle=True)

label_mapping = {
    'AD': 0,
    'CN': 1,
}
mapping_to_label = {v: k for k, v in label_mapping.items()}

# Load and preprocess data
Images, Labels = load_and_preprocess_data()

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

with open('test_accuracy.txt', 'r') as file:
    accuracy = float(file.read())

print(f'The accuracy of Model is: {accuracy}%')

# Load and preprocess data
Images, Labels = load_and_preprocess_data()

# Split data into training and testing sets
train_indices, test_indices = train_test_split(list(range(Images.shape[0])), train_size=0.8, test_size=0.2, shuffle=True)
x_train = Images[train_indices]
y_train = Labels[train_indices]
x_test = Images[test_indices]
y_test = Labels[test_indices]

ad_indices = random.sample(np.where(y_test == 0)[0].tolist(), 5)
cn_indices = random.sample(np.where(y_test == 1)[0].tolist(), 5)
sample_indices = ad_indices + cn_indices

sample_images = x_test[sample_indices]
sample_labels = y_test[sample_indices]
sample_predictions = model.predict(sample_images)
sample_predicted_labels = np.argmax(sample_predictions, axis=1)

for i, (image, actual_label, predicted_labels) in enumerate(zip(sample_images, sample_labels, sample_predicted_labels)):
    plt.subplot(2, 5, i + 1)
    plt.imshow(image.astype("uint8"))

    if isinstance(actual_label, np.ndarray):
        actual_label_names = [mapping_to_label[label] for label in actual_label]
        actual_label_display = ', '.join(actual_label_names)
    else:
        actual_label_display = mapping_to_label[actual_label]

    predicted_label_names = [mapping_to_label[predicted_labels]]
    
    plt.title(f"Actual: {actual_label_display}\nPredicted: {', '.join(predicted_label_names)}")
    plt.axis("off")
plt.show()