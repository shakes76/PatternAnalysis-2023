import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate

def unet_model(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)
    
    # Downsampling (Context) Path
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    
    # Upsampling (Localization) Path
    u1 = UpSampling2D((2, 2))(c3)
    concat1 = Concatenate()([u1, c2])  # Concatenate with feature maps from the downsampling path
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(concat1)
    u2 = UpSampling2D((2, 2))(c4)
    concat2 = Concatenate()([u2, c1])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat2)
    
    # Output Layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)
    
    model = tf.keras.Model(inputs, outputs)
    return model
