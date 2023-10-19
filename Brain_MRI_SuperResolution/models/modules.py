from keras.layers import Input, Conv2D, BatchNormalization, Add, LeakyReLU, Lambda
from keras.layers import Conv2DTranspose
from keras.models import Model
import tensorflow as tf

# Setting TensorFlow to execute functions eagerly for easier debugging and prototyping
tf.config.run_functions_eagerly(True)


def residual_block(x, filters, kernel_size=3, use_batch_norm=True):
    """
    Define a residual block for the Super-Resolution CNN.

    :param x: Input tensor
    :param filters: Number of filters for the convolutional layers
    :param kernel_size: Size of the convolutional kernel
    :param use_batch_norm: Boolean to decide whether to use batch normalization
    :return: Tensor after passing through the residual block
    """

    # Saving the input for later use in the shortcut connection
    shortcut = x
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = LeakyReLU()(x)

    # Optional batch normalization
    if use_batch_norm:
        x = BatchNormalization()(x)

    # Second convolutional layer
    x = Conv2D(filters, kernel_size, padding='same')(x)

    # Optional batch normalization
    if use_batch_norm:
        x = BatchNormalization()(x)

    # Adding the input tensor (shortcut connection) to the result
    x = Add()([shortcut, x])
    x = LeakyReLU()(x)
    return x


def sub_pixel_cnn(input_shape):
    """
    Define the Super-Resolution Convolutional Neural Network (SRCNN) model.

    :param input_shape: Shape of the low-resolution input image
    :return: Keras Model that represents the SRCNN
    """

    # Input layer
    inputs = Input(shape=input_shape)

    # Initial convolution block
    x = Conv2D(64, (5, 5), padding='same')(inputs)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    # Add a residual block
    x = residual_block(x, 64)

    # Up-scaling block
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Lambda(lambda x: tf.nn.depth_to_space(x, 2))(x)  # Sub-pixel convolution

    # Final block
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same')(x)

    return Model(inputs=inputs, outputs=x)
