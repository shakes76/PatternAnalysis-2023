import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from modules import improved_unet

# The height and width of the processed image
img_height = img_width = 256


def dice_sim_coef(y_true, y_pred, epsilon=1.0):
    axes = tuple(range(1, len(y_pred.shape) - 1))
    numerator = 2. * tf.math.reduce_sum(y_pred * y_true, axes)
    denominator = tf.math.reduce_sum(tf.math.square(y_pred) + tf.math.square(y_true), axes)
    return tf.reduce_mean((numerator + epsilon) / (denominator + epsilon))


def dice_sim_coef_loss(y_true, y_pred):
    return 1 - dice_sim_coef(y_true, y_pred)


def initialise_model():
    # Creates the improved UNet model
    unet_model = improved_unet(img_height, img_width, 3)
    # Sets the training parameters for the model
    unet_model.compile(optimizer='adam', loss=[dice_sim_coef_loss],
                       metrics=[dice_sim_coef])
    # Prints a summary of the model compiled
    unet_model.summary()
    # Plots a summary of the model's architecture
    #tf.keras.utils.plot_model(unet_model, show_shapes=True)
    # Moves the model.png file created to the Results folder. If model.png is already present in the Results
    # sub directory, it is deleted and replaced with the new model.png
    # if os.path.exists(os.getcwd() + "\Results\model.png"):
    #     os.remove(os.getcwd() + "\Results\model.png")
    # os.rename(os.getcwd() + "\model.png", os.getcwd() + "\Results\model.png")
    return unet_model