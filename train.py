import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from modules import improved_unet
from dataset import process_images, create_ds, img_height, img_width

"""Model Saving Constants"""
# If a pre-trained model should be used
use_saved_model = False
# If the model being trained should be saved (Note: Only works if use_saved_model = True)
save_model = True

"""Model Training Constants"""
# If the images should be shuffled (Note: Masks and their related image are not changed)
shuffle = True
# The dataset split percentage for the training dataset
training_split = 0.8
# The dataset split percentage for the validation dataset
# (Note: The testing dataset will be the remaining dataset once the training and validation datasets have been taken)
validation_split = 0.1
# The shuffle size to be used
shuffle_size = 50

# The height and width of the processed image
img_height = img_width = 256
# The batch size to be used
batch_size = 16
# The number of training epochs
epochs = 10
# The number of times a similar validation dice coefficient score is achieved before training is stopped early
patience = 5

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
    tf.keras.utils.plot_model(unet_model, show_shapes=True)
    # Moves the model.png file created to the Results folder. If model.png is already present in the Results
    # sub directory, it is deleted and replaced with the new model.png
    if os.path.exists(os.getcwd() + "\Results\model.png"):
        os.remove(os.getcwd() + "\Results\model.png")
    os.rename(os.getcwd() + "\model.png", os.getcwd() + "\Results\model.png")
    return unet_model

