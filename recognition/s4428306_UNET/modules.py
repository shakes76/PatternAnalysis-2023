#Code in this section is based on code from:
#https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/
#Papers used to write this:
#[1] Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge (arXiv:1802.10508)
#[2] CNN-based Segmentation of Medical Imaging Data (arXiv:1701.03056)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LeakyReLU

def context_module(x, n_filters):
    """
    A context module to be used in the context pathway, as described in [1].
    """
    og_x = x
    x = layers.Conv2D(n_filters, 3, padding="same", activation=LeakyReLU(0.01), kernel_initializer="he_normal")(x)
    x = layers.Conv2D(n_filters, 3, padding="same", activation=LeakyReLU(0.01), kernel_initializer="he_normal")(x)
    x = layers.Dropout(0.3)(x)
    return x + og_x

def upsampling_module(x, n_filters):
    """
    An upsampling module to be used in the localization pathway as described in [1].
    """
    x = layers.UpSampling2D(size=(2,2))(x)
    x = layers.Conv2D(n_filters, 3, padding="same", activation=LeakyReLU(0.01), kernel_initializer="he_normal")(x)
    return x

def localization_module(x, conv_features, n_filters):
    """
    A localization module to be used in the localization pathway as described in [1].
    """
    x = layers.concatenate([x, conv_features])
    x = layers.Conv2D(n_filters, 3, padding="same", activation=LeakyReLU(0.01), kernel_initializer="he_normal")(x)
    x = layers.Conv2D(n_filters, 1, padding="same", activation=LeakyReLU(0.01), kernel_initializer="he_normal")(x)
    return x

def segmentation_layer(x, n_filters):
    """
    A segmentation layer to be used in the localization pathway as described in [2].
    """
    x = layers.Conv2D(n_filters, 1, padding="same", activation=LeakyReLU(0.01), kernel_initializer="he_normal")(x)
    return x

def build_improved_unet_model():
    """
    Builds the improved U-Net model to be compiled and trained.
    """
    inputs = layers.Input(shape=(128,128,3))
    #The context pathway.
    x = layers.Conv2D(16, 3, padding="same", activation=LeakyReLU(0.01), kernel_initializer="he_normal")(inputs)
    c1 = context_module(x, 16)
    x = layers.Conv2D(32, 3, padding="same", strides=2, activation=LeakyReLU(0.01), kernel_initializer="he_normal")(c1)
    c2 = context_module(x, 32)
    x = layers.Conv2D(64, 3, padding="same", strides=2, activation=LeakyReLU(0.01), kernel_initializer="he_normal")(c2)
    c3 = context_module(x, 64)
    x = layers.Conv2D(128, 3, padding="same", strides=2, activation=LeakyReLU(0.01), kernel_initializer="he_normal")(c3)
    c4 = context_module(x, 128)
    x = layers.Conv2D(256, 3, padding="same", strides=2, activation=LeakyReLU(0.01), kernel_initializer="he_normal")(c4)
    x = context_module(x, 256)
    #The localization pathway.
    x = upsampling_module(x, 128)
    x = localization_module(x, c4, 128)
    x = upsampling_module(x, 64)
    l1 = localization_module(x, c3, 64)
    x = upsampling_module(l1, 32)
    l2 = localization_module(x, c2, 32)
    x = upsampling_module(l2, 16)
    x = layers.concatenate([x, c1])
    x = layers.Conv2D(32, 3, padding="same", activation=LeakyReLU(0.01), kernel_initializer="he_normal")(x)
    #Upscale and add segmentation layers.
    x = segmentation_layer(x, 1)
    s1 = segmentation_layer(l1, 1)
    s1 = layers.UpSampling2D(size=(2,2))(s1)
    s2 = segmentation_layer(l2, 1) + s1
    s2 = layers.UpSampling2D(size=(2,2))(s2)
    x = x + s2
    outputs = tf.keras.activations.sigmoid(x)
    return tf.keras.Model(inputs, outputs)

