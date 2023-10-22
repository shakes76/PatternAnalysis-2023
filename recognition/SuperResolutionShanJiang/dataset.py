import tensorflow as tf
import os
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
upscale_factor = 4 # ratio that dowansample orginal image for training and upscale images to predict at
input_height_size = crop_height_size // upscale_factor
input_width_size = crop_width_size // upscale_factor
batch_size = 8

#Specify directory containing training dataset
training_dir = "D:/temporary_workspace/comp3710_project/PatternAnalysis_2023_Shan_Jiang/recognition/SuperResolutionShanJiang/train_dataset"

#Create traning dataset
train_ds = image_dataset_from_directory(
    training_dir,
    batch_size=batch_size,
    image_size=(crop_height_size, crop_width_size),
    validation_split=0.2,
    subset="training",
    seed=1337,
    label_mode=None,
)

#Create validation dataset
valid_ds = image_dataset_from_directory(
    training_dir,
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

# A fucntion that turns given image to grey scale and crop it 
def process_input(input,input_height_size,input_width_size):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return tf.image.resize(y, [input_height_size, input_width_size], method="area")

# A fucntion that turn given image to grey scale
def process_target(input):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return y


# Process train dataset:create low resolution images and corresponding high resolution images, and put the pair into a tuple
train_ds = train_ds.map(
    lambda x: (process_input(x, input_height_size, input_width_size), process_target(x))
)
train_ds = train_ds.prefetch(buffer_size=32)

# Process validation dataset:create low resolution images and corresponding high resolution images, and put the pair into a tuple
valid_ds = valid_ds.map(
    lambda x: (process_input(x, input_height_size, input_width_size), process_target(x))
)
valid_ds = valid_ds.prefetch(buffer_size=32)

#Specify directory containing testing dataset
test_path = 'D:/temporary_workspace/comp3710_project/PatternAnalysis_2023_Shan_Jiang/recognition/SuperResolutionShanJiang/test_dataset'
#Put path of each testing image into a sorted list
test_img_paths = sorted(
    [
        os.path.join(test_path, fname)
        for fname in os.listdir(test_path)
        if fname.endswith(".jpeg")
    ]
)

#return a list containing path of each image for testing
def get_test_img_paths():
    return test_img_paths

#Specify directory containing prediction dataset
prediction_path = "D:/temporary_workspace/comp3710_project/PatternAnalysis_2023_Shan_Jiang/recognition/SuperResolutionShanJiang/prediction_dataset"
#Put path of each prediction image into a sorted list
prediction_path = sorted(
    [
        os.path.join(prediction_path, fname)
        for fname in os.listdir(prediction_path)
        if fname.endswith(".jpeg")
    ]
)


# return a list containing path of each image to be predicted
def get_prediction_img_paths():
    return prediction_path



