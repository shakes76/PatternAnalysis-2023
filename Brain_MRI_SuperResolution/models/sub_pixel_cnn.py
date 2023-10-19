from keras.layers import Input, Conv2D, BatchNormalization, Add, LeakyReLU, Lambda
from keras.layers import Conv2DTranspose
from keras.models import Model
import tensorflow as tf

tf.config.run_functions_eagerly(True)


def residual_block(x, filters, kernel_size=3, use_batch_norm=True):
    shortcut = x
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = LeakyReLU()(x)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = LeakyReLU()(x)
    return x


def sub_pixel_cnn(input_shape):
    inputs = Input(shape=input_shape)

    # Initial convolution block
    x = Conv2D(64, (5, 5), padding='same')(inputs)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    # Add a residual block
    x = residual_block(x, 64)

    # Upscaling block
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Lambda(lambda x: tf.nn.depth_to_space(x, 2))(x)  # Sub-pixel convolution

    # Final block
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same')(x)

    return Model(inputs=inputs, outputs=x)
