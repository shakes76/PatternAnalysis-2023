#Code in this section is based on code from:
#https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def context_module(x, n_filters):
    #TODO: Write specification.
    #NOTE: Layers in paper are said to be 3D, but this may be because they incorporate channels.
    #      Also unsure about whether padding should be used, and what type.
    og_x = x
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Dropout(0.3)(x)
    return x + og_x

#NOTE: For the report, could talk about how these are used instead of transposed convs (as mentioned in the paper),
#      and do some experiments with these replaced.
def upsampling_module(x, n_filters):
    #TODO: Write specification.
    #NOTE: This looks like what is mentioned in the paper, but the arguments to the upsample layers, or whether they
    #      should be 3D may require some experimentation.
    x = layers.UpSampling2D(size=(2,2))(x)
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    return x

def localization_module(x, n_filters):
    #TODO: Write specifictaion.
    #NOTE: Again, unsure if these should be 2D or 3D.
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Conv2D(n_filters, 1, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    return x

#NOTE: From what I can see from the papers this just a 1x1 (or 1x1x1) convolution.
#      Going to define it like this for readability.
def segmentation_layer(x, n_filters):
    #TODO: Write specifictaion.
    #NOTE: Again, unsure if these should be 2D or 3D.
    x = layers.Conv2D(n_filters, 1, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    return x

