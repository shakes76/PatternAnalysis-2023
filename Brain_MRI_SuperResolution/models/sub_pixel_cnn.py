from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Lambda, BatchNormalization, Add


def sub_pixel_cnn(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (5, 5), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x1 = BatchNormalization()(x1)

    # Introducing skip connection
    x2 = Add()([x, x1])
    x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)

    # Another layer
    x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(x2)
    x3 = BatchNormalization()(x3)

    x4 = Add()([x2, x3])
    x4 = Conv2D(4, (3, 3), activation='relu', padding='same')(x4)
    x4 = BatchNormalization()(x4)

    def subpixel(x):
        import tensorflow as tf
        return tf.nn.depth_to_space(x, 2)

    outputs = Lambda(subpixel)(x4)

    return Model(inputs=inputs, outputs=outputs)
