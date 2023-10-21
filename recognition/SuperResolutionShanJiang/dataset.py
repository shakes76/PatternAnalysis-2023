import tensorflow as tf
import os

import tensorflow as tf
import math


from tensorflow import keras
from keras import layers
from keras.utils import load_img
from keras.utils import array_to_img
from keras.utils import img_to_array
from keras.preprocessing import image_dataset_from_directory

from IPython.display import display





#Set parameters for cropping
crop_width_size = 256
crop_height_size = 248
upscale_factor = 4
input_height_size = crop_height_size // upscale_factor
input_width_size = crop_width_size // upscale_factor
batch_size = 8
data_dir = 'D:/temporary_workspace/comp3710_project/PatternAnalysis_2023_Shan_Jiang/recognition/SuperResolutionShanJiang/original/train/AD'



#Create traning dataset
train_ds = image_dataset_from_directory(
    data_dir,
    batch_size=batch_size,
    image_size=(crop_height_size, crop_width_size),
    validation_split=0.2,
    subset="training",
    seed=1337,
    label_mode=None,
)

#Create validation dataset
valid_ds = image_dataset_from_directory(
    data_dir,
    batch_size=batch_size,
    image_size=(crop_height_size, crop_width_size),
    validation_split=0.2,
    subset="validation",
    seed=1337,
    label_mode=None,
)

# resacla training and validation images to take values in the range [0, 1].
def scaling(input_image):
    input_image = input_image / 255.0
    return input_image

train_ds = train_ds.map(scaling)
valid_ds = valid_ds.map(scaling)

# for batch in train_ds.take(1):
#     for img in batch:
#         display(array_to_img(img))




def process_input(input,input_height_size,input_width_size):
    """ turn given image to grey scale and crop it 

    Args:
        input: image to be processed
        input_width_size: width to be cropped into
        input_height_size: height to be cropped into

    Returns:
        tensor: processed image
    """
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return tf.image.resize(y, [input_height_size, input_width_size], method="area")


def process_target(input):
    """turn given image to grey scale

    Args:
        input: image to be processed

    Returns:
        tensor: processed image
    """
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return y


# Process train dataset:create low resolution images and corresponding high resolution images
train_ds = train_ds.map(
    lambda x: (process_input(x, input_height_size, input_width_size), process_target(x))
)
train_ds = train_ds.prefetch(buffer_size=32)

# Process validation dataset:create low resolution images and corresponding high resolution images
valid_ds = valid_ds.map(
    lambda x: (process_input(x, input_height_size, input_width_size), process_target(x))
)
valid_ds = valid_ds.prefetch(buffer_size=32)

# for batch in train_ds.take(1):
#     for img in batch[0]:
#         display(array_to_img(img))
#     for img in batch[1]:
#         display(array_to_img(img))




