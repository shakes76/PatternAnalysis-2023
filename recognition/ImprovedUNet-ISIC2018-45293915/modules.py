import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
import tensorflow_addons as tfa
# 'tensorflow-addons' is an officially supported repository implementing new functionality:
# More info at https://www.tensorflow.org/addons. Version 0.9.1 is required for TF 2.1.
# TFA allows for a InstanceNormalization layer (rather than a BatchNormalization layer), as was implemented in the
# referenced 'improved UNet'. This layer is necessary due to the usage of my small batch-size of 2, which can lead to
# "stochasticity induced ...[which]... may destabilize batch normalizaton" -
#   F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation and
#   Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,” Feb. 2018. [Online]. Available:
#   https://arxiv.org/abs/1802.10508v1.
# While BatchNormalization normalises across the batch, InstanceNormalization normalises each batch separately.

# --------------------------------------------
# GLOBAL CONSTANTS
# --------------------------------------------

LEAKY_RELU_ALPHA = 0.01
DROPOUT = 0.35
L2_WEIGHT_DECAY = 0.0005
CONV_PROPERTIES = dict(
    kernel_regularizer=l2(L2_WEIGHT_DECAY),
    bias_regularizer=l2(L2_WEIGHT_DECAY),
    padding="same")
I_NORMALIZATION_PROPERTIES = dict(
    axis=3,
    center=True,
    scale=True,
    beta_initializer="random_uniform",
    gamma_initializer="random_uniform")


# --------------------------------------------
# IMPROVED UNET MODEL FOR ISICS BINARY SEGMENTATION
# --------------------------------------------

# Implementation based off the 'improved UNet': https://arxiv.org/abs/1802.10508v1.
# 2D implementation rather than 3D as 2D inputs/outputs are required.
def improved_unet(width, height, channels):
    input_layer = keras.Input(shape=(width, height, channels))

    conv1 = keras.layers.Conv2D(16, (3, 3), input_shape=(width, height, channels), **CONV_PROPERTIES)(input_layer)
    norm1 = tfa.layers.InstanceNormalization(**I_NORMALIZATION_PROPERTIES)(conv1)
    relu1 = keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(norm1)
    context1 = context_module(relu1, 16)
    add1 = keras.layers.Add()([conv1, context1])

    conv2 = keras.layers.Conv2D(32, (3, 3), strides=2, **CONV_PROPERTIES)(add1)
    norm2 = tfa.layers.InstanceNormalization(**I_NORMALIZATION_PROPERTIES)(conv2)
    relu2 = keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(norm2)
    context2 = context_module(relu2, 32)
    add2 = keras.layers.Add()([relu2, context2])

    conv3 = keras.layers.Conv2D(64, (3, 3), strides=2, **CONV_PROPERTIES)(add2)
    norm3 = tfa.layers.InstanceNormalization(**I_NORMALIZATION_PROPERTIES)(conv3)
    relu3 = keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(norm3)
    context3 = context_module(relu3, 64)
    add3 = keras.layers.Add()([relu3, context3])

    conv4 = keras.layers.Conv2D(128, (3, 3), strides=2, **CONV_PROPERTIES)(add3)
    norm4 = tfa.layers.InstanceNormalization(**I_NORMALIZATION_PROPERTIES)(conv4)
    relu4 = keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(norm4)
    context4 = context_module(relu4, 128)
    add4 = keras.layers.Add()([relu4, context4])

    conv5 = keras.layers.Conv2D(256, (3, 3), strides=2, **CONV_PROPERTIES)(add4)
    norm5 = tfa.layers.InstanceNormalization(**I_NORMALIZATION_PROPERTIES)(conv5)
    relu5 = keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(norm5)
    context5 = context_module(relu5, 256)
    add5 = keras.layers.Add()([relu5, context5])
    upsample1 = upsampling_module(add5, 128)

    concat1 = keras.layers.Concatenate()([add4, upsample1])
    localization1 = localisation_module(concat1, 128)
    upsample2 = upsampling_module(localization1, 64)

    concat2 = keras.layers.Concatenate()([add3, upsample2])
    localization2 = localisation_module(concat2, 64)
    upsample3 = upsampling_module(localization2, 32)

    concat3 = keras.layers.Concatenate()([add2, upsample3])
    localization3 = localisation_module(concat3, 32)
    upsample4 = upsampling_module(localization3, 16)

    concat4 = keras.layers.Concatenate()([add1, upsample4])
    conv6 = keras.layers.Conv2D(32, (3, 3), **CONV_PROPERTIES)(concat4)
    norm6 = tfa.layers.InstanceNormalization(**I_NORMALIZATION_PROPERTIES)(conv6)
    relu6 = keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(norm6)

    seg_layer1 = keras.layers.Activation('sigmoid')(localization2)
    upsample5 = upsampling_module(seg_layer1, 32)
    seg_layer2 = keras.layers.Activation('sigmoid')(localization3)
    sum1 = keras.layers.Add()([upsample5, seg_layer2])
    upsample6 = upsampling_module(sum1, 32)
    seg_layer3 = keras.layers.Activation('sigmoid')(relu6)
    sum2 = keras.layers.Add()([upsample6, seg_layer3])

    output_layer = keras.layers.Conv2D(1, (1, 1), activation="sigmoid", **CONV_PROPERTIES)(sum2)
    u_net = keras.Model(inputs=[input_layer], outputs=[output_layer])
    return u_net


# --------------------------------------------
# MODULES
# --------------------------------------------

# A 'Context Module', based off the 'improved UNet'.
def context_module(input_layer, out_filter):
    conv1 = keras.layers.Conv2D(out_filter, (3, 3), **CONV_PROPERTIES)(input_layer)
    # bn1 = keras.layers.BatchNormalization()(conv1)
    norm1 = tfa.layers.InstanceNormalization(**I_NORMALIZATION_PROPERTIES)(conv1)
    relu1 = keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(norm1)
    dropout1 = keras.layers.Dropout(DROPOUT)(relu1)
    conv2 = keras.layers.Conv2D(out_filter, (3, 3), **CONV_PROPERTIES)(dropout1)
    # bn2 = keras.layers.BatchNormalization()(conv2)
    norm2 = tfa.layers.InstanceNormalization(**I_NORMALIZATION_PROPERTIES)(conv2)
    relu2 = keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(norm2)
    return relu2

# An 'Upsampling Module', based off the 'improved UNet'.
def upsampling_module(input_layer, out_filter):
    upsampled = keras.layers.UpSampling2D(size=(2, 2))(input_layer)
    conv = keras.layers.Conv2D(out_filter, (3, 3), **CONV_PROPERTIES)(upsampled)
    # bn = keras.layers.BatchNormalization()(conv)
    norm = tfa.layers.InstanceNormalization(**I_NORMALIZATION_PROPERTIES)(conv)
    relu = keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(norm)
    return relu

# A 'Localisation Module', based off the 'improved UNet'.
def localisation_module(input_layer, out_filter):
    conv1 = keras.layers.Conv2D(out_filter, (3, 3), **CONV_PROPERTIES)(input_layer)
    # bn1 = keras.layers.BatchNormalization()(conv1)
    norm1 = tfa.layers.InstanceNormalization(**I_NORMALIZATION_PROPERTIES)(conv1)
    relu1 = keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(norm1)
    conv2 = keras.layers.Conv2D(out_filter, (1, 1), **CONV_PROPERTIES)(relu1)
    # bn2 = keras.layers.BatchNormalization()(conv2)
    norm2 = tfa.layers.InstanceNormalization(**I_NORMALIZATION_PROPERTIES)(conv2)
    relu2 = keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(norm2)
    return relu2