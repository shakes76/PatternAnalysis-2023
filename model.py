import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the Efficient Sub-Pixel CNN model
def subpixel_conv2d(input_tensor, scale, num_filters):
    x = layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same')(input_tensor)
    x = layers.UpSampling2D(size=(scale, scale))(x)
    return x

def efficient_subpixel_cnn(input_shape, scale):
    # Input layer
    input_lr = keras.Input(shape=input_shape)

    # Sub-pixel convolution blocks
    x = subpixel_conv2d(input_lr, scale, 64)
    x = subpixel_conv2d(x, scale, 64)
    x = subpixel_conv2d(x, scale, 64)

    # Output layer
    output_hr = layers.Conv2D(1, (3, 3), activation='relu', padding='same')(x)

    # Create the Keras model
    model = keras.Model(inputs=input_lr, outputs=output_hr, name='efficient_subpixel_cnn')

    return model

# Define the input shape and upscaling factor
input_shape = (64, 64, 1)  # Modify this according to your dataset's input size
scale = 4  # Adjust the scale factor based on your super-resolution task

# Create the model
model = efficient_subpixel_cnn(input_shape, scale)

# Compile the model with an appropriate loss function and optimizer
model.compile(optimizer='adam', loss='mean_squared_error')

# Summary of the model architecture
model.summary()
