#Code in this section is based on code from:
#https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/
#TODO: Reference both papers. (Improved UNet and the segmentation layer one.)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LeakyReLU

def context_module(x, n_filters):
    #TODO: Write specification.
    #NOTE: Layers in paper are said to be 3D, but this may be because they incorporate channels.
    #      Leaving padding as "same" to prevent headaches with dimensions.
    og_x = x
    x = layers.Conv2D(n_filters, 3, padding="same", activation=LeakyReLU(0.01), kernel_initializer="he_normal")(x)
    x = layers.Conv2D(n_filters, 3, padding="same", activation=LeakyReLU(0.01), kernel_initializer="he_normal")(x)
    x = layers.Dropout(0.3)(x)
    return x + og_x

#NOTE: For the report, could talk about how these are used instead of transposed convs (as mentioned in the paper),
#      and do some experiments with these replaced.
def upsampling_module(x, n_filters):
    #TODO: Write specification.
    #NOTE: This looks like what is mentioned in the paper, but the arguments to the upsample layers, or whether they
    #      should be 3D may require some experimentation.
    x = layers.UpSampling2D(size=(2,2))(x)
    x = layers.Conv2D(n_filters, 3, padding="same", activation=LeakyReLU(0.01), kernel_initializer="he_normal")(x)
    return x

#NOTE: Concatenation happens inside this module now, because it always happens before data is passed into it.
#      Remember that there is one more concatenation before one of the final convolutions.
def localization_module(x, conv_features, n_filters):
    #TODO: Write specifictaion.
    #NOTE: Again, unsure if these should be 2D or 3D.
    x = layers.concatenate([x, conv_features])
    x = layers.Conv2D(n_filters, 3, padding="same", activation=LeakyReLU(0.01), kernel_initializer="he_normal")(x)
    x = layers.Conv2D(n_filters, 1, padding="same", activation=LeakyReLU(0.01), kernel_initializer="he_normal")(x)
    return x

#NOTE: From what I can see from the papers this is just a 1x1 (or 1x1x1) convolution.
#      Should the papers be referenced somewhere?
#      Going to define it like this for readability.
def segmentation_layer(x, n_filters):
    #TODO: Write specifictaion.
    #NOTE: Again, unsure if these should be 2D or 3D.
    x = layers.Conv2D(n_filters, 1, padding="same", activation=LeakyReLU(0.01), kernel_initializer="he_normal")(x)
    return x

def build_improved_unet_model():
    #TODO: Write specificatoin.
    #NOTE: The extra dimension in the training data must be something wrong with the batching.
    inputs = layers.Input(shape=(128,128,3))
    #TODO: Name this path. (down?)
    #NOTE: Again, unsure if these should be 2D or 3D.
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
    #TODO: Name this path. (up?)
    x = upsampling_module(x, 128)
    x = localization_module(x, c4, 128)
    x = upsampling_module(x, 64)
    l1 = localization_module(x, c3, 64)
    x = upsampling_module(l1, 32)
    l2 = localization_module(x, c2, 32)
    x = upsampling_module(l2, 16)
    x = layers.concatenate([x, c1])
    x = layers.Conv2D(32, 3, padding="same", activation=LeakyReLU(0.01), kernel_initializer="he_normal")(x)
    #NOTE: Unsure how many filters segmentation layers should have.
    #      It looks like they all need to be the same for them to add together.
    #      Setting them all to 1 so that the output matches the masks.
    #      Changed filters to 2, because there are 2 classes.
    x = segmentation_layer(x, 2)
    #Upscale and add segmentation layers.
    s1 = segmentation_layer(l1, 2)
    s1 = layers.UpSampling2D(size=(2,2))(s1)
    s2 = segmentation_layer(l2, 2) + s1
    s2 = layers.UpSampling2D(size=(2,2))(s2)
    x = x + s2
    outputs = layers.Softmax(axis=3)(x)
    #NOTE: Squeezing to make sure dimensions are the same as the masks.
    #      Looks like this actually shouldn't be done.
    #outputs = tf.squeeze(outputs)
    #NOTE: A different model will need to be returned once the new class is set up and training is written.
    return tf.keras.Model(inputs, outputs)

#TODO: Write new model and training function.
