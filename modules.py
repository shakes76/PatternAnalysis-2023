import tensorflow as tf
from tensorflow.keras.layers import concatenate, Conv2D, BatchNormalization, Dropout, Input, LeakyReLU, MaxPooling2D, UpSampling2D


def improved_unet(height, width, channels):    

    """Constants"""
    # The number of filters for each convolutional layer
    fil = 16
    # The kernel size to use
    kern = (3, 3)
    # The padding argument used for each convolutional layer
    pad = 'same'
    # The dropout rate used by each Dropout layer
    drop = 0.3
    # The alpha rate used by each LeakyReLU layer
    alp = 0.01
    # The activation type for the convolutional layers
    actv = 'relu'

    # Building the model
    inputs = Input((height, width, channels))

    # 3x3x3 convolution
    conv1 = Conv2D(fil, kern, padding=pad, activation=actv)(inputs)
    # Instance Normalisation
    bat1 = BatchNormalization()(conv1)
    # Leaky ReLU
    relu1 = LeakyReLU(alpha=alp)(bat1)
    # Context module
    context1 = Conv2D(fil, kern, padding=pad, activation=actv)(relu1)
    context1 = Dropout(drop)(context1)
    context1 = Conv2D(fil, kern, padding=pad, activation=actv)(context1)
    # Instance Normalisation
    bat1 = BatchNormalization()(context1)
    # Leaky ReLU
    relu1 = LeakyReLU(alpha=alp)(bat1)
    # Element-wise sum
    out1 = conv1 + relu1

    # Down-sampling
    down_samp2 = MaxPooling2D()(out1)
    # 3x3x3 stride 2 convolution
    conv2 = Conv2D(fil * 2, kern, padding=pad, activation=actv)(down_samp2)
    # Instance Normalisation
    bat2 = BatchNormalization()(conv2)
    # Leaky ReLU
    relu2 = LeakyReLU(alpha=alp)(bat2)
    # Context module
    context2 = Conv2D(fil * 2, kern, padding=pad, activation=actv)(relu2)
    context2 = Dropout(drop)(context2)
    context2 = Conv2D(fil * 2, kern, padding=pad, activation=actv)(context2)
    # Instance Normalisation
    bat2 = BatchNormalization()(context2)
    # Leaky ReLU
    relu2 = LeakyReLU(alpha=alp)(bat2)
    # Element-wise sum
    out2 = conv2 + relu2

    # Down-sampling
    down_samp3 = MaxPooling2D()(out2)
    # 3x3x3 stride 2 convolution
    conv3 = Conv2D(fil * 4, kern, padding=pad, activation=actv)(down_samp3)
    # Instance Normalisation
    bat3 = BatchNormalization()(conv3)
    # Leaky ReLU
    relu3 = LeakyReLU(alpha=alp)(bat3)
    # Context module
    context3 = Conv2D(fil * 4, kern, padding=pad, activation=actv)(relu3)
    context3 = Dropout(drop)(context3)
    context3 = Conv2D(fil * 4, kern, padding=pad, activation=actv)(context3)
    # Instance Normalisation
    bat3 = BatchNormalization()(context3)
    # Leaky ReLU
    relu3 = LeakyReLU(alpha=alp)(bat3)
    # Element-wise sum
    out3 = conv3 + relu3

    # Down-sampling
    down_samp4 = MaxPooling2D()(out3)
    # 3x3x3 stride 2 convolution
    conv4 = Conv2D(fil * 8, kern, padding=pad, activation=actv)(down_samp4)
    # Instance Normalisation
    bat4 = BatchNormalization()(conv4)
    # Leaky ReLU
    relu4 = LeakyReLU(alpha=alp)(bat4)
    # Context module
    context4 = Conv2D(fil * 8, kern, padding=pad, activation=actv)(relu4)
    context4 = Dropout(drop)(context4)
    context4 = Conv2D(fil * 8, kern, padding=pad, activation=actv)(context4)
    # Instance Normalisation
    bat4 = BatchNormalization()(context4)
    # Leaky ReLU
    relu4 = LeakyReLU(alpha=alp)(bat4)
    # Element-wise sum
    out4 = conv4 + relu4

    # Down-sampling
    down_samp5 = MaxPooling2D()(out4)
    # 3x3x3 stride 2 convolution
    conv5 = Conv2D(fil * 16, kern, padding=pad, activation=actv)(down_samp5)
    # Instance Normalisation
    bat5 = BatchNormalization()(conv5)
    # Leaky ReLU
    relu5 = LeakyReLU(alpha=alp)(bat5)
    # Context module
    context5 = Conv2D(fil * 16, kern, padding=pad, activation=actv)(relu5)
    context5 = Dropout(drop)(context5)
    context5 = Conv2D(fil * 16, kern, padding=pad, activation=actv)(context5)
    # Instance Normalisation
    bat5 = BatchNormalization()(context5)
    # Leaky ReLU
    relu5 = LeakyReLU(alpha=alp)(bat5)
    # Element-wise sum
    out5 = conv5 + relu5
    # Up-sampling
    up_samp5 = UpSampling2D(size=(2, 2))(out5)
    up_samp5 = Conv2D(fil * 8, kern, padding=pad, activation=actv)(up_samp5)