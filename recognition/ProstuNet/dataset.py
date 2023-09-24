# dataset.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

class Data:
    """
    A class to represent a Data model.

    ...

    Attributes
    ----------
    p : type
        info
        
    Methods
    -------
    func(additional=""):
        info about function
    """

    def __init__(self, p):
        """
        Constructs all the necessary attributes for the U-Net object.

        Parameters
        ----------
            p : type
                info
        """

        self.p = p

    def resize(self, input_image, input_mask):
        ''' Resizesimages and masks to 128.
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
