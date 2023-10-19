import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

"""
Loads the ADNI dataset using ImageDataGenerators and performs trainsforms on data

Outputs:
    train_generator - the training set generator, split into AD/NC classes
    labels - the training set generator, split into AD/NC classes
"""
def load_dataset(path):
    train_datagen = ImageDataGenerator(rescale=1.0/255.0,  # Normalize pixel values
                                rotation_range=20,   # Ratates each image by up to 20 degrees
                                width_shift_range=0.2, # Changes the width of each image
                                height_shift_range=0.2, # Changes the height of each image
                                horizontal_flip=True, # Randomly flips each image horizontally
                                vertical_flip=True)  # Use 80% of the data for training) # Randomly flips each image vertically
    
    train_generator = train_datagen.flow_from_directory(path, # Path for train set
                                                target_size=(256, 240),  # Set the desired image size
                                                batch_size=1, # Det the batch size
                                                class_mode='categorical')  # Split into AD and NC classes
    

    test_datagen = ImageDataGenerator(rescale=1.0/255.0) # Normalize pixel values

    test_generator = test_datagen.flow_from_directory("AD_NC/test", # Path for test set
                                                    target_size=(256, 240), # Det the desired image size
                                                    batch_size=1, # Set the batch size
                                                    class_mode='categorical') # Split into AD and NC classes
    return train_generator, test_generator

"""
Loads a variable number of data points from a data generator.
Done as training the classifier on the whole generator would take too
much computation resources. 

Inputs:
    data_generator - the data generator (train or test)
    num - the number of data points to read
"""
def load_data_classifier(data_generator, num):
    anchor = []
    labels = []

    for i in range(num):
        anchor_img, anchor_label = data_generator[i]
        anchor.append(anchor_img[0]) 
        labels.append(tf.math.argmax(anchor_label, axis=1).numpy()) 

    anchor = tf.convert_to_tensor(anchor)
    labels = tf.concat(labels, axis=0)
    return [anchor, labels]


"""
Python function used to create anchor, positive and negative triplets for the siamese neural network. 
Two triples are returned, the image data and the image labels. 

Input:
    data_generator - the data_generator (either train or test)
    num - the number of (anchor, positive, negative) triples 

Outputs:
    [anchor_triplets, pos_triplets, neg_triplets] - triplet array used for input into the siamese model
    [anchor_labels, pos_labels, neg_labels] - triplet array of labels 
"""
def load_data_triplets(data_generator, num):
    anchor_triplets = []
    pos_triplets = []
    neg_triplets = []
    anchor_labels = []
    pos_labels = []
    neg_labels = []

    for i in range(num):
        anchor_img, anchor_label = data_generator[i]
        anchor_triplets.append(anchor_img[0])
        anchor_labels.append(tf.math.argmax(anchor_label, axis=1).numpy()) 

        # check random data until same class as anchor
        while True:
            pos_idx = random.randint(0, len(data_generator) - 1)
            pos_img, pos_label = data_generator[pos_idx]
            if pos_label.argmax() == anchor_label.argmax():
                pos_triplets.append(pos_img[0])
                pos_labels.append(tf.math.argmax(pos_label, axis=1).numpy())
                break

        # check random data until opposite class as anchor
        while True:
            neg_idx = random.randint(0, len(data_generator) - 1)
            neg_img, neg_label = data_generator[neg_idx]

            if neg_label.argmax() != anchor_label.argmax():
                neg_triplets.append(neg_img[0])
                neg_labels.append(tf.math.argmax(neg_label, axis=1).numpy())
                break

    anchor_triplets = tf.convert_to_tensor(anchor_triplets)
    pos_triplets = tf.convert_to_tensor(pos_triplets)
    neg_triplets = tf.convert_to_tensor(neg_triplets)
    anchor_labels = tf.concat(anchor_labels, axis=0)
    pos_labels = tf.concat(pos_labels, axis=0)
    neg_labels = tf.concat(neg_labels, axis=0)

    return [anchor_triplets, pos_triplets, neg_triplets], [anchor_labels, pos_labels, neg_labels]