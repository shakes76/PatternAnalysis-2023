import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
import tensorflow_addons as tfa
# TensorFlow Addons (tfa) is used here because it provides support for Instance Normalization (IN)
# as opposed to Batch Normalization (BN). This choice aligns with the findings from the referenced paper
# (https://arxiv.org/abs/1802.10508v1), where IN is preferred over BN due to the potential destabilization
# of batch normalization caused by small batch sizes in the training process.

# --------------------------------------------
# GLOBAL CONSTANTS
# --------------------------------------------

LEAKY_RELU_ALPHA = 0.01 # Alpha value for LeakyReLU activation function
DROPOUT = 0.35 # Dropout rate
L2_WEIGHT_DECAY = 0.0005 # L2 weight decay
CONV_PROP = dict(
    kernel_regularizer=l2(L2_WEIGHT_DECAY),
    bias_regularizer=l2(L2_WEIGHT_DECAY),
    padding="same")
IN_PROP = dict(
    axis=3,
    center=True,
    scale=True,
    beta_initializer="random_uniform",
    gamma_initializer="random_uniform")


def improved_unet(width, height, channels):
    # Improved UNet Model for ISIC Segmentation
    # -------------------------------------------------

    # Implementation based on the 'improved UNet' architecture from https://arxiv.org/abs/1802.10508v1.
    # This 2D implementation is designed for image segmentation tasks, with instance normalization (IN)
    # for improved stability, especially with small batch sizes.

    # Architecture Overview:
    #
    # [Input] -> [Encoder] -> [Decoder] -> [Output]
    #
    # Key Components:
    # - Encoder: Down-sampling path with convolutional layers and context modules.
    # - Decoder: Up-sampling path with localization and upsampling modules.
    # - Context Module: Enhances feature extraction and learning by adding context.
    # - Upsampling Module: Upscales feature maps to recover spatial information.
    # - Localisation Module: Fine-tunes feature maps for better segmentation.
    #
    # Output Layer:
    # - Sigmoid activation for binary segmentation.
    #
    # Detailed Architecture:
    # [Input] -> [Conv] -> [IN] -> [LeakyReLU] -> [Context] -> [Add] -> [Conv] -> [IN] -> [LeakyReLU] -> ...
    # ... -> [Upsampling] -> [Conv] -> [IN] -> [LeakyReLU] -> [Add] -> [Concatenate] -> [Localization] -> ...
    # ... -> [Upsampling] -> [Conv] -> [IN] -> [LeakyReLU] -> [Add] -> [Concatenate] -> [Localization] -> ...
    # ... -> [Upsampling] -> [Conv] -> [IN] -> [LeakyReLU] -> [Add] -> [Concatenate] -> [Localization] -> ...
    # ... -> [Upsampling] -> [Conv] -> [IN] -> [LeakyReLU] -> [Add] -> [Concatenate] -> [Conv] -> [IN] -> ...
    # ... [LeakyReLU] -> [Sigmoid] -> [Upsampling] -> [Sigmoid] -> [Add] -> [Upsampling] -> [Sigmoid] -> [Add] -> [Output]

    input_layer = keras.Input(shape=(width, height, channels))

    # Encoder
    # -------------------------------------------------

    conv1 = keras.layers.Conv2D(16, (3, 3), input_shape=(width, height, channels), **CONV_PROP)(input_layer)
    norm1 = tfa.layers.InstanceNormalization(**IN_PROP)(conv1)
    relu1 = keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(norm1)
    context1 = context_module(relu1, 16)
    add1 = keras.layers.Add()([conv1, context1])

    conv2 = keras.layers.Conv2D(32, (3, 3), strides=2, **CONV_PROP)(add1)
    norm2 = tfa.layers.InstanceNormalization(**IN_PROP)(conv2)
    relu2 = keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(norm2)
    context2 = context_module(relu2, 32)
    add2 = keras.layers.Add()([relu2, context2])

    conv3 = keras.layers.Conv2D(64, (3, 3), strides=2, **CONV_PROP)(add2)
    norm3 = tfa.layers.InstanceNormalization(**IN_PROP)(conv3)
    relu3 = keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(norm3)
    context3 = context_module(relu3, 64)
    add3 = keras.layers.Add()([relu3, context3])

    conv4 = keras.layers.Conv2D(128, (3, 3), strides=2, **CONV_PROP)(add3)
    norm4 = tfa.layers.InstanceNormalization(**IN_PROP)(conv4)
    relu4 = keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(norm4)
    context4 = context_module(relu4, 128)
    add4 = keras.layers.Add()([relu4, context4])

    # Decoder
    # -------------------------------------------------

    conv5 = keras.layers.Conv2D(256, (3, 3), strides=2, **CONV_PROP)(add4)
    norm5 = tfa.layers.InstanceNormalization(**IN_PROP)(conv5)
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
    conv6 = keras.layers.Conv2D(32, (3, 3), **CONV_PROP)(concat4)
    norm6 = tfa.layers.InstanceNormalization(**IN_PROP)(conv6)
    relu6 = keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(norm6)

    seg_layer1 = keras.layers.Activation('sigmoid')(localization2)
    upsample5 = upsampling_module(seg_layer1, 32)
    seg_layer2 = keras.layers.Activation('sigmoid')(localization3)
    sum1 = keras.layers.Add()([upsample5, seg_layer2])
    upsample6 = upsampling_module(sum1, 32)
    seg_layer3 = keras.layers.Activation('sigmoid')(relu6)
    sum2 = keras.layers.Add()([upsample6, seg_layer3])

    # Output
    # -------------------------------------------------

    output_layer = keras.layers.Conv2D(1, (1, 1), activation="sigmoid", **CONV_PROP)(sum2)
    u_net = keras.Model(inputs=[input_layer], outputs=[output_layer])
    return u_net



# MODULES
# -------------------------------------------------

# Context Module based inspired by the Improved UNet paper.
def context_module(input_layer, out_filter):
    conv1 = keras.layers.Conv2D(out_filter, (3, 3), **CONV_PROP)(input_layer)
    norm1 = tfa.layers.InstanceNormalization(**IN_PROP)(conv1)
    relu1 = keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(norm1)
    dropout1 = keras.layers.Dropout(DROPOUT)(relu1)
    conv2 = keras.layers.Conv2D(out_filter, (3, 3), **CONV_PROP)(dropout1)
    norm2 = tfa.layers.InstanceNormalization(**IN_PROP)(conv2)
    relu2 = keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(norm2)
    return relu2

# Upsampling Module based inspired by the Improved UNet paper.
def upsampling_module(input_layer, out_filter):
    upsampled = keras.layers.UpSampling2D(size=(2, 2))(input_layer)
    conv = keras.layers.Conv2D(out_filter, (3, 3), **CONV_PROP)(upsampled)
    norm = tfa.layers.InstanceNormalization(**IN_PROP)(conv)
    relu = keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(norm)
    return relu

# Localisation Module inspired by the Improved UNet paper.
def localisation_module(input_layer, out_filter):
    conv1 = keras.layers.Conv2D(out_filter, (3, 3), **CONV_PROP)(input_layer)
    norm1 = tfa.layers.InstanceNormalization(**IN_PROP)(conv1)
    relu1 = keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(norm1)
    conv2 = keras.layers.Conv2D(out_filter, (1, 1), **CONV_PROP)(relu1)
    norm2 = tfa.layers.InstanceNormalization(**IN_PROP)(conv2)
    relu2 = keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(norm2)
    return relu2