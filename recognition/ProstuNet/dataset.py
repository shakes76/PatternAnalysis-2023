# dataset.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import torch
import nibabel as nib
import random
import os
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import glob
from PIL import Image

class Data:
    """
    A class to represent a Data model.

    ...

    Methods
    -------
    open_data(self, imagePath, maskPath):
        Opens data from image and mask path and creates a tensor representation

    load_image_train(self, datapoint):
        Loads image data set for training and performs preprocessing.

    load_image_test(self, datapoint):
        Loads image data set for testing and performs preprocessing.

    resize(self, input_image, input_mask):
        Resizesimages and masks to 128x128

    flip(self, input_image, input_mask):
        flips images and masks randomly.
    
    normalize(self, input_image, input_mask):
        Rescaling real-valued images and masks into a compareable range 
    
    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the Data model.
        """
        # import data here 
        self.dataset = self.open_data("data/semantic_MRs_anon/*", 
                      "data/semantic_labels_anon/*")
        # split the dataset
        self.train, self.validation, self.test = torch.utils.data.random_split(self.dataset, [179, 16, 16])
        # build input pipeline with data using map()
        self.train_dataset = self.train.map(self.load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
        self.test_dataset = self.test.map(self.load_image_test, num_parallel_calls=tf.data.AUTOTUNE)

        # create batches
        BATCH_SIZE = 64
        BUFFER_SIZE = 1000
        self.train_batches = self.train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
        self.train_batches = self.train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.validation_batches = self.test_dataset.take(3000).batch(BATCH_SIZE)
        self.test_batches = self.test_dataset.skip(3000).take(669).batch(BATCH_SIZE)

        # sample display:
        self.sample_batch = next(iter(self.train_batches))
        self.random_index = np.random.choice(self.sample_batch[0].shape[0])
        self.sample_image, self.sample_mask = self.sample_batch[0][self.random_index], self.sample_batch[1][self.random_index]
        self.display([self.sample_image, self.sample_mask])

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx): 
        ''' Given an item index, uses the nib label library to open nii files
            Parameters:
                idx: image index
            
            Returns:
                image and mask at index'''
        image_p = self.inputs[idx]
        mask_p = self.masks[idx]

        image = nib.load(image_p)
        image = np.asarray(image.dataobj)

        mask = nib.load(mask_p)
        mask = np.asarray(mask.dataobj)
        
        image = self.totensor(image)
        image = image.unsqueeze(0)
        image = image.data

        mask = self.totensor(mask)
        mask = mask.unsqueeze(0)
        mask = mask.data
        
        return image, mask

    def open_data(self, imagePath, maskPath):
        ''' Opens data from image and mask path and creates a tensor representation
            Parameters:
                imagePath: path to images
                maskPath: path to mask'''
        self.inputs = []
        self.masks = []
        #retrieve path from dataset
        for f in sorted(glob.iglob(image_path)): 
            self.inputs.append(f)
        for f in sorted(glob.iglob(mask_path)):
            self.masks.append(f)
        self.totensor = transforms.ToTensor()

    def load_image_train(self, datapoint):
        ''' Loads image data set for training and performs preprocessing.
            Augmentation is performed only on the training set.
            Parameters:
                datapoint: data subset for training
            
            Returns:
                training data'''
        input_image = datapoint["image"]
        input_mask = datapoint["segmentation_mask"]
        input_image, input_mask = self.resize(input_image, input_mask)
        input_image, input_mask = self.augment(input_image, input_mask)
        input_image, input_mask = self.normalize(input_image, input_mask)
        return input_image, input_mask
    
    def load_image_test(self, datapoint):
        ''' Loads image data set for testing and performs preprocessing.
            Parameters:
                datapoint: data subset for testing
            
            Returns:
                testing data'''
        input_image = datapoint["image"]
        input_mask = datapoint["segmentation_mask"]
        input_image, input_mask = self.resize(input_image, input_mask)
        input_image, input_mask = self.normalize(input_image, input_mask)
        return input_image, input_mask
    


    def resize(self, input_image, input_mask):
        ''' Resizesimages and masks to 128x128.
            Parameters:
                input_image: image
                input_mask: mask
            
            Returns:
                resized image and mask'''
        input_image = tf.image.resize(input_image, (128, 128), method="nearest")
        input_mask = tf.image.resize(input_mask, (128, 128), method="nearest")
        return input_image, input_mask

    def flip(self, input_image, input_mask):
        ''' flips images and masks randomly.
            Parameters:
            input_image: image
            input_mask: mask
            
            Returns:
                flipped image and mask'''
        if tf.random.uniform(()) > 0.5:
        # Random flipping of the image and mask
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)
        return input_image, input_mask



    def normalize(self, input_image, input_mask):
        ''' Rescaling real-valued images and masks into a compareable range 
            Parameters:
                input_image: image
                input_mask: mask
        
            Returns:
                normalised images and masks'''
        input_image = tf.cast(input_image, tf.float32) / 255.0
        input_mask -= 1
        return input_image, input_mask

    def display(self, display_list):
        ''' Display images 
            Parameters:
                display_list: list of images and masks to display'''

        plt.figure(figsize=(15, 15))
        title = ["Input Image", "True Mask", "Predicted Mask"]
        for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
        plt.show()

