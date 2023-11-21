import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, UpSampling2D, Input, LeakyReLU, Add, Dropout, \
    BatchNormalization
from tensorflow.keras.models import Model


def context_module(inputs, filters):
    """
    Context module is a pre-activation residual block with two 3x3x3 convolutional layers
    and a dropout layer (drop = 0.3) in between.
    Reference: https://arxiv.org/abs/1802.10508v1
    :param inputs: an initial input / 3x3x3 convolutional layer
    :param filters: filter size as per the diagram in above reference Fig. 1
    :return:
    """
    ctxt = BatchNormalization()(inputs)
    ctxt = Conv2D(filters, (3, 3), strides=1, padding='same', activation=LeakyReLU(alpha=0.01))(ctxt)
    ctxt = Dropout(0.3)(ctxt)
    ctxt = BatchNormalization()(ctxt)
    ctxt = Conv2D(filters, (3, 3), strides=1, padding='same', activation=LeakyReLU(alpha=0.01))(ctxt)
    return ctxt


def context_layer(inputs, stride, filters):
    """
    An entire layer of the model in the context (down) section. It is composed of a 3x3x3 Convolution layer, followed
    by a context module. The Element-wise sum of these are then calculated and returned as output of the function.
    Reference: https://arxiv.org/abs/1802.10508v1
    :param inputs: the previous layer / input to build from
    :param stride: stride size as per the diagram in above reference Fig. 1
    :param filters: filter size as per the diagram in above reference Fig. 1
    :return: element-wise sum of the Conv layer and Context Module.
    """
    # 3x3x3 Conv
    c = Conv2D(filters, (3, 3), strides=stride, padding='same', activation=LeakyReLU(alpha=0.01))(inputs)

    # Context Module
    ctxt1 = context_module(c, filters)

    # Element-wise sum
    return Add()([c, ctxt1])


def upsampling_module(inputs, add, filters):
    """
    Upsampling Module as per reference below to upscale then half the number of feature maps.
    Reference: https://arxiv.org/abs/1802.10508v1
    :param inputs: previous layer to be built onto
    :param add: previous element-wise sum result from the same level of the Context section
    :param filters: filter size as per the diagram in above reference Fig. 1
    :return: Concatenation of the upsampled result and the add parameter
    """
    up = UpSampling2D((2, 2))(inputs)
    up = Conv2D(filters, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01))(up)

    return Concatenate()([up, add])


def localization_module(inputs, filters):
    """
    Localization Module (as per reference below) to half the number of feature maps.
    Reference: https://arxiv.org/abs/1802.10508v1
    :param inputs: previous layer to be built onto
    :param filters: filter size as per the diagram in above reference Fig. 1
    :return:
    """
    up = Conv2D(filters, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01))(inputs)
    up = Conv2D(filters, (1, 1), padding='same', activation=LeakyReLU(alpha=0.01))(up)
    return up


def improved_unet():
    """
    Function to create the Improved UNet model to be used during the training phase.
    """
    # ---- Inputs ----
    inputs = tf.keras.layers.Input((256, 256, 1))  # 1 input channel for greyscale images

    # ---- DOWN 1 (16 filter) ----
    add1 = context_layer(inputs, 1, 16)  # stride of 1 for the initial 3x3x3 conv

    # ---- DOWN 2 (32 filter) ----
    add2 = context_layer(add1, 2, 32)

    # ---- DOWN 3 (64 filter) ----
    add3 = context_layer(add2, 2, 64)

    # ---- DOWN 4 (128 filter) ----
    add4 = context_layer(add3, 2, 128)

    # ---- DOWN 5 (256 filter) ----
    add5 = context_layer(add4, 2, 256)

    # ---- UP 1 (128 filter) ----
    concat1 = upsampling_module(add5, add4, 128)
    u1 = localization_module(concat1, 128)

    # ---- UP 2 (64 filter) ----
    concat2 = upsampling_module(u1, add3, 64)
    u2 = localization_module(concat2, 64)

    # ---- Segmentation ----
    seg1 = Conv2D(3, (1, 1), strides=1, padding="same")(u2)
    seg1 = UpSampling2D((2, 2))(seg1)

    # ---- UP 3 (32 filter) ----
    concat3 = upsampling_module(u2, add2, 32)
    u3 = localization_module(concat3, 32)

    # ---- Segmentation ----
    seg2 = Conv2D(3, (1, 1), strides=1, padding="same")(u3)
    seg2 = Add()([seg1, seg2])
    seg2 = UpSampling2D((2, 2))(seg2)

    # ---- UP 4 (16 filter) ----
    concat4 = upsampling_module(u3, add1, 16)

    # ---- UP 5 (32 filter) ----
    # 3x3x3 conv with stride 2
    u5 = Conv2D(32, (3, 3), strides=1, padding='same')(concat4)

    # ---- Segmentation ----
    seg3 = Conv2D(3, (1, 1), strides=1, padding="same")(u5)
    seg3 = Add()([seg2, seg3])

    # ---- Outputs ----
    outputs = Conv2D(2, (1, 1), activation='softmax')(seg3)
    model = Model(inputs, outputs)

    return model
