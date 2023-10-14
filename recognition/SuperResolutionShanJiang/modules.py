import tensorflow as tf
from tensorflow import keras
from keras import layers

def get_model(upscale_factor=4, channels=1):
    """build a super-resolution model

    Args:
        upscale_factor: ratio to upscale the image. Defaults to 3.
        channels: Number of channels. Defaults to 1.

    Returns:
        keras.Model: super-resolution model
    """
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = keras.Input(shape=(None, None, channels))
    x = layers.Conv2D(64, 5, **conv_args)(inputs)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(32, 3, **conv_args)(x)
    x = layers.Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

    return keras.Model(inputs, outputs)
