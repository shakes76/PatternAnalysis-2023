import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model, load_model, save_model
from keras.layers.convolutional import Input, Activation, BatchNormalization, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
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


def encoder_block(inp, filters, padding='same', pool_stride=2, activation='relu'):
    """
    Encoder block of a UNet passes the result from the convolution block
    above to a max pooling layer
    """
    x = conv_block(inp, filters, padding, activation)
    p = MaxPooling2D(pool_size=(2, 2), strides=pool_stride)(x)
    return x, p

# Functions to build the decoder block
def decoder_block(inp, filters, concat_layer, padding='same'):
    # Upsample the feature maps
    x = Conv2DTranspose(filters, (2,2), strides=(2,2), padding=padding)(inp)
    x = concatenate([x, concat_layer]) # Concatenation/Skip conncetion with conjuagte encoder
    x = conv_block(x, filters) # Passed into the convolution block above
    return x

class UNet2D(input, latent_dim=64, activation=None, kernel=[3, 3], channels=1, name_prefix=''):
    # Encoder
    inputs = Input((256,256,1))
    d1,p1 = encoder_block(inputs,64)
    d2,p2 = encoder_block(p1,128)
    d3,p3 = encoder_block(p2,256)
    d4,p4 = encoder_block(p3,512) 

    # Middle convolution block (no max pooling) - final output will now be upsampled
    mid = conv_block(p4,1024) #Midsection   

    # Decoder
    e2 = decoder_block(mid,512,d4) #Conjugate of encoder 4
    e3 = decoder_block(e2,256,d3) #Conjugate of encoder 3
    e4 = decoder_block(e3,128,d2) #Conjugate of encoder 2 
    e5 = decoder_block(e4,64,d1) #Conjugate of encoder 1
    outputs = Conv2D(1, (1,1), activation=None)(e5) #Final Output
    
    ml = Model(inputs=[inputs], outputs=[outputs, o1], name='Unet')  
    