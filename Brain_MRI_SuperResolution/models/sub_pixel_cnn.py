from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Add, LeakyReLU, Dropout, Lambda
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Reshape

from tensorflow.keras.models import Model
import tensorflow as tf

def residual_block(y, nb_channels, _strides=(1, 1)):
    shortcut = y

    # First layer
    y = Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
    y = LeakyReLU()(y)
    y = BatchNormalization()(y)

    # Second layer
    y = Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
    y = BatchNormalization()(y)

    # Add shortcut value to main path
    y = Add()([shortcut, y])
    y = LeakyReLU()(y)

    return y
def sub_pixel_cnn(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (5, 5), padding='same')(inputs)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    # Add multiple residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    # Dropout for regularization
    x = Dropout(0.3)(x)

    x = Conv2D(4, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    # Sub-pixel convolution to get 200x200
    x = Lambda(lambda x: tf.nn.depth_to_space(x, 2))(x)

    # Additional upscaling to get 300x300
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same')(x) # This will upscale from 200 to 400
    x = Lambda(lambda img: tf.image.resize(img, [300,300]))(x)

    outputs = x

    return Model(inputs=inputs, outputs=outputs)
