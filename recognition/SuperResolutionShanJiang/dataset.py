import tensorflow as tf
import os

import tensorflow as tf

import os
import math
import numpy as np

from tensorflow import keras
from keras import layers
from keras.utils import load_img
from keras.utils import array_to_img
from keras.utils import img_to_array
from keras.preprocessing import image_dataset_from_directory

from IPython.display import display



# def downsize_data(input_directory,output_directory):
#     """
#     Downsample image data in input directory and save to output directory
#     """
#     # Define the size for downsampling (factor of 4)
#     downsample_factor = 4

#     # Get a list of all JPEG files in the input directory
#     jpeg_files = [f for f in os.listdir(input_directory)]

#     # Loop through the JPEG files, resize, and save
#     for file_name in jpeg_files:
#         input_path = os.path.join(input_directory, file_name)
#         output_path = os.path.join(output_directory, file_name)

#         # Read the image from the file
#         image = tf.io.read_file(input_path)
#         image = tf.image.decode_jpeg(image, channels=3)  # Decode the image

#         # Resize the image by a factor of 4
#         new_height = tf.shape(image)[0] // downsample_factor
#         new_width = tf.shape(image)[1] // downsample_factor
#         resized_image = tf.image.resize(image, (new_height, new_width), method=tf.image.ResizeMethod.BILINEAR, antialias=True)

#         # Cast the tensor to uint8 before encoding as JPEG
#         resized_image = tf.cast(resized_image, tf.uint8)

#         # Encode and save the downsized image as a JPEG
#         tf.io.write_file(output_path, tf.image.encode_jpeg(resized_image).numpy())
                                                                                                                                                        
#     print("Downsampling complete.")

# # Down size data
# # input_directory = 'D:/temporary_workspace/comp3710_project/PatternAnalysis_2023_Shan_Jiang/recognition/SuperResolutionShanJiang/original/test/AD'
# # output_directory = 'D:/temporary_workspace/comp3710_project/PatternAnalysis_2023_Shan_Jiang/recognition/SuperResolutionShanJiang/downsized/test/AD'
# # downsize_data(input_directory,output_directory)

# input_directory = 'D:/temporary_workspace/comp3710_project/PatternAnalysis_2023_Shan_Jiang/recognition/SuperResolutionShanJiang/original/test/NC'
# output_directory = 'D:/temporary_workspace/comp3710_project/PatternAnalysis_2023_Shan_Jiang/recognition/SuperResolutionShanJiang/downsized/test/NC'
# downsize_data(input_directory,output_directory)

# input_directory = 'D:/temporary_workspace/comp3710_project/PatternAnalysis_2023_Shan_Jiang/recognition/SuperResolutionShanJiang/original/train/AD'
# output_directory = 'D:/temporary_workspace/comp3710_project/PatternAnalysis_2023_Shan_Jiang/recognition/SuperResolutionShanJiang/downsized/train/AD'
# downsize_data(input_directory,output_directory)

# input_directory = 'D:/temporary_workspace/comp3710_project/PatternAnalysis_2023_Shan_Jiang/recognition/SuperResolutionShanJiang/original/train/NC'
# output_directory = 'D:/temporary_workspace/comp3710_project/PatternAnalysis_2023_Shan_Jiang/recognition/SuperResolutionShanJiang/downsized/train/NC'
# downsize_data(input_directory,output_directory)

#Set parameters for cropping
crop_width_size_ = 256
crop_height_size_ = 249
upscale_factor = 4
input_width_size = crop_width_size_ // upscale_factor
input_height_size = crop_height_size_ // upscale_factor
batch_size = 8
data_dir = 'D:/temporary_workspace/comp3710_project/PatternAnalysis_2023_Shan_Jiang/recognition/SuperResolutionShanJiang/original/train/AD'

#Create traning dataset
train_ds = image_dataset_from_directory(
    data_dir,
    batch_size=batch_size,
    image_size=(input_height_size, input_width_size),
    validation_split=0.2,
    subset="training",
    seed=1337,
    label_mode=None,
)

#Create validation dataset
valid_ds = image_dataset_from_directory(
    data_dir,
    batch_size=batch_size,
    image_size=(input_height_size, input_width_size),
    validation_split=0.2,
    subset="validation",
    seed=1337,
    label_mode=None,
)

