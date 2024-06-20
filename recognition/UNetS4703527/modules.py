from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model

def encoder_block(inputs, num_filters):
    """
    Create an encoder block in the U-Net.

    Parameters:
    inputs (Tensor): The input tensor.
    num_filters (int): The number of filters for the convolutional layers.

    Returns:
    Tuple: Output tensor after the encoder block and the pooled features.
    """
    # Conv layer
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # Second conv layer
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # Max Pooling layer
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    """
    Create a decoder block in the U-Net.

    Paramters:
    inputs (Tensor): The input tensor.
    skip_features (Tensor): The features from the encoder block.
    num_filters (int): The number of filters for the convolutional layers.

    Returns:
    Tensor: Output tensor after the decoder block.
    """
    # Transposed convolution layer (up conv)
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    # Copy and crop
    x = Concatenate()([x, skip_features])
    # Conv layer
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # Second Conv layer
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def Unet(input_shape):
    """
    Create the U-Net.

    Paramters:
    input_shape (tuple): The shape of the input data.

    Returns:
    Model: The U-Net model.
    """
    inputs = Input(input_shape)
    # 4 Encoder Blocks
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    
    # Conv bridging
    x = Conv2D(1024, 3, padding="same")(p4)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(1024, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # 4 Decoder blocks
    d1 = decoder_block(x, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # Conv 1x1
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs)
    return model