import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

"""
Loads the ADNI dataset using ImageDataGenerators and performs trainsforms on data



Outputs:
    train_generator - the training set generator, split into AD/NC classes
    test_generator - the training set generator, split into AD/NC classes
"""
def load_dataset():
    train_datagen = ImageDataGenerator(rescale=1.0/255.0, 
                                rotation_range=20,   
                                width_shift_range=0.2, 
                                height_shift_range=0.2, 
                                horizontal_flip=True, 
                                vertical_flip=True) 
    
    train_generator = train_datagen.flow_from_directory("AD_NC/train", # Path for train set, might need to change
                                                target_size=(256, 240),  
                                                batch_size=1, 
                                                class_mode='categorical') 
    

    test_datagen = ImageDataGenerator(rescale=1.0/255.0) 

    test_generator = test_datagen.flow_from_directory("AD_NC/test", # Path for test set, might need to change
                                                    target_size=(256, 240), 
                                                    batch_size=1,
                                                    class_mode='categorical')
    return train_generator, test_generator

"""
Loads a variable number of data points from a data generator.
Done as training the classifier on the whole generator would take too
much computation resources. 

Inputs:
    anchor - the data generator (train or test)
    num - the number of data points to read

Outpuse:
    anchor - num many anchor image data 
    labels - the labels associated with each index
"""
def load_data_classifier(data_generator, num):
    anchor = []
    labels = []

    for i in range(num):
        anchor_img, anchor_label = data_generator[i]
        anchor.append(anchor_img[0]) 
        labels.append(anchor_label[0])
        
    anchor = tf.convert_to_tensor(anchor)
    labels = tf.convert_to_tensor(labels)
    return anchor, labels


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
        anchor_labels.append(anchor_label[0][1])

        # check random data until same class as anchor
        while True:
            pos_idx = random.randint(0, len(data_generator) - 1)
            pos_img, pos_label = data_generator[pos_idx]
            if pos_label.argmax() == anchor_label.argmax():
                pos_triplets.append(pos_img[0])
                pos_labels.append(pos_label[0][1])
                break

        # check random data until opposite class as anchor
        while True:
            neg_idx = random.randint(0, len(data_generator) - 1)
            neg_img, neg_label = data_generator[neg_idx]

            if neg_label.argmax() != anchor_label.argmax():
                neg_triplets.append(neg_img[0])
                neg_labels.append(neg_label[0][1])
                break

    anchor_triplets = tf.convert_to_tensor(anchor_triplets)
    pos_triplets = tf.convert_to_tensor(pos_triplets)
    neg_triplets = tf.convert_to_tensor(neg_triplets)
    anchor_labels = tf.convert_to_tensor(anchor_labels)
    pos_labels = tf.convert_to_tensor(pos_labels)
    neg_labels = tf.convert_to_tensor(neg_labels)

    return [anchor_triplets, pos_triplets, neg_triplets], [anchor_labels, pos_labels, neg_labels]
