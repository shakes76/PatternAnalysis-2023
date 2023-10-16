import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers.convolutional import Input, Activation, BatchNormalization, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

# Functions to build the encoder path (source: https://towardsdatascience.com/image-segmentation-unet-and-deep-supervision-loss-using-keras-model-f21a9856750a)
def conv_block(inp, filters, padding='same', activation='relu'):
    """
    Convolution block of a UNet encoder
    """
    x = Conv2D(filters, (3, 3), padding=padding, activation=activation)(inp)
    x = Conv2D(filters, (3, 3), padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation)(x)
    return x


def encoder_block(inp, filters, padding='same', pool_stride=2,
                  activation='relu'):
    """
    Encoder block of a UNet passes the result from the convolution block
    above to a max pooling layer
    """
    x = conv_block(inp, filters, padding, activation)
    p = MaxPooling2D(pool_size=(2, 2), strides=pool_stride)(x)
    return x, p

class UNet2D(input, latent_dim=64, activation=None, kernel=[3, 3], channels=1, name_prefix=''):
    # encoder
    # building the first block
    inputs = Input((256,256,1))
    d1,p1 = encoder_block(inputs,64)

    # building the other four
    d2,p2 = encoder_block(p1,128)
    d3,p3 = encoder_block(p2,256)
    d4,p4 = encoder_block(p3,512) 

    # Middle convolution block (no max pooling) - final output will now be upsampled
    mid = conv_block(p4,1024) #Midsection     
    