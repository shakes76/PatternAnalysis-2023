from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, Dropout
from tensorflow.keras.models import Model

def conv_block(inputs, num_filters, dropout_rate=0.3):
    """
    Create a convolutional block consisting of two convolutional layers with dropout.

    Parameters:
    - inputs: Input tensor.
    - num_filters (int): Number of filters in the convolutional layers.
    - dropout_rate (float): Dropout rate for the dropout layer.

    Returns:
    - Tensor: Output tensor after the convolutional block.
    """
    x = Conv2D(num_filters, 3, padding="same", activation='relu')(inputs)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(num_filters, 3, padding="same", activation='relu')(x)
    return x

def encoder_block(inputs, num_filters, dropout_rate=0.3):
    """
    Create an encoder block consisting of a convolutional block followed by max pooling.

    Parameters:
    - inputs: Input tensor.
    - num_filters (int): Number of filters in the convolutional layers.
    - dropout_rate (float): Dropout rate for the dropout layer.

    Returns:
    - Tuple: Two tensors - the output tensor of the convolutional block and the max-pooled tensor.
    """
    x = conv_block(inputs, num_filters, dropout_rate)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters, dropout_rate=0.3):
    """
    Create a decoder block consisting of a transposed convolutional layer and concatenation with skip connections.

    Parameters:
    - inputs: Input tensor.
    - skip_features: Skip connection tensor from the encoder block.
    - num_filters (int): Number of filters in the transposed convolutional layer.
    - dropout_rate (float): Dropout rate for the dropout layer.

    Returns:
    - Tensor: Output tensor after the decoder block.
    """
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters, dropout_rate)
    return x

def build_unet(input_shape, dropout_rate=0.3):
    """
    Build a U-Net model architecture.

    Parameters:
    - input_shape (tuple): Shape of the input tensor (height, width, channels).
    - dropout_rate (float): Dropout rate for the dropout layers.

    Returns:
    - Model: U-Net model.
    """
    inputs = Input(input_shape)

    """ Encoder """
    s1, p1 = encoder_block(inputs, 64, dropout_rate)
    s2, p2 = encoder_block(p1, 128, dropout_rate)
    s3, p3 = encoder_block(p2, 256, dropout_rate)
    s4, p4 = encoder_block(p3, 512, dropout_rate)

    b1 = conv_block(p4, 1024, dropout_rate)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512, dropout_rate)
    d2 = decoder_block(d1, s3, 256, dropout_rate)
    d3 = decoder_block(d2, s2, 128, dropout_rate)
    d4 = decoder_block(d3, s1, 64, dropout_rate)

    """ Outputs """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    model = Model(inputs, outputs)
    return model
