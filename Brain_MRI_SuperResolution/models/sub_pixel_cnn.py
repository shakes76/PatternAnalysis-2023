from keras.models import Model
from keras.layers import Input, Conv2D
import tensorflow as tf;
from torchvision.transforms import Lambda


def sub_pixel_cnn(scale_factor=4, channels=1):
    inputs = Input(shape=(None, None, channels))

    # Add layers according to Efficient Sub-Pixel CNN
    net = Conv2D(filters=channels * (scale_factor ** 2), kernel_size=3, padding='same')(inputs)
    # Use a depth_to_space function (or similar) to shuffle and upscale.
    outputs = Lambda(lambda x: tf.nn.depth_to_space(x, scale_factor))(net)

    return Model(inputs=inputs, outputs=outputs)
