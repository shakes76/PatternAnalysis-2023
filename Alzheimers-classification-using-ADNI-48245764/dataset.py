import tensorflow as tf
import numpy as np
import os
from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    """
    Load and preprocess image data for training and testing.

    This function reads image files, preprocesses them, and splits the data into training,
    validation, and testing sets.

    Returns:
        Tuple containing the following:
        - train_images: Training images (as TensorFlow tensors)
        - test_images: Testing images (as TensorFlow tensors)
        - validation_images: Validation images (as TensorFlow tensors)
        - train_labels: Labels for the training images
        - test_labels: Labels for the testing images
        - validation_labels: Labels for the validation images
    """
    
    training_images = 'AD_NC/train'
    testing_images = 'AD_NC/test'

    # List image files
    AD_train_files = os.listdir(os.path.join(training_images, 'AD'))
    AD_test_files = os.listdir(os.path.join(testing_images, 'AD'))
    NC_train_files = os.listdir(os.path.join(training_images, 'NC'))
    NC_test_files = os.listdir(os.path.join(testing_images, 'NC'))

    training_data = [] 

    i = 0
    # Load and preprocess AD training images
    while i < len(AD_train_files):
        filename = AD_train_files[i]
        image = load_img(os.path.join(training_images, 'AD', filename), target_size=(140, 140, 3))
        image = img_to_array(image)
        training_data.append([image, 1])
        i += 1

    
    i = 0
    # Load and preprocess NC training images
    while i < len(NC_train_files):
        filename = NC_train_files[i]
        image = load_img(os.path.join(training_images, 'NC', filename), target_size=(140, 140, 3))
        image = img_to_array(image)
        training_data.append([image, 0])
        i += 1

    training_images = []
    training_labels = []

    
    i = 0
    # Split training data into images and labels
    while i < len(training_data):
        training_images.append(training_data[i][0])
        training_labels.append(training_data[i][1])
        i += 1

    # Split training data into training and validation sets
    trainset, valid, trainsetlabel, valid_label = train_test_split(training_images, training_labels, train_size=0.80, shuffle=True)

    train_images = tf.convert_to_tensor(np.array(trainset, dtype=np.uint8))
    train_images = tf.cast(train_images, tf.float16) / 255.0
    train_labels = tf.convert_to_tensor(trainsetlabel)

    validation_images = tf.convert_to_tensor(np.array(valid, dtype=np.uint8))
    validation_images = tf.cast(validation_images, tf.float16) / 255.0
    validation_labels = tf.convert_to_tensor(valid_label)

    testing_data = []

    
    i = 0
    # Load and preprocess AD testing images
    while i < len(AD_test_files):
        filename = AD_test_files[i]
        image = load_img(os.path.join(testing_images, 'AD', filename), target_size=(140, 140, 3))
        image = img_to_array(image)
        testing_data.append([image, 1])
        i += 1

    i = 0
    # Load and preprocess NC testing images
    while i < len(NC_test_files):
        filename = NC_test_files[i]
        image = load_img(os.path.join(testing_images, 'NC', filename), target_size=(140, 140, 3))
        image = img_to_array(image)
        testing_data.append([image, 0])
        i += 1

    testing_images = []
    testing_labels = []

    
    i = 0
    # Split testing data into images and labels
    while i < len(testing_data):
        testing_images.append(testing_data[i][0])
        testing_labels.append(testing_data[i][1])
        i += 1

    test_images = tf.convert_to_tensor(np.array(testing_images, dtype=np.uint8))
    test_images = tf.cast(test_images, tf.float16) / 255.0
    test_labels = tf.convert_to_tensor(testing_labels)

    return train_images, test_images, validation_images, train_labels, test_labels, validation_labels
