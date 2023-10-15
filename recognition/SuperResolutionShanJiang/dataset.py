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
crop_width_size = 256
crop_height_size = 248
upscale_factor = 4
input_height_size = crop_height_size // upscale_factor
input_width_size = crop_width_size // upscale_factor
batch_size = 8
# data_dir = 'D:/temporary_workspace/comp3710_project/PatternAnalysis_2023_Shan_Jiang/recognition/SuperResolutionShanJiang/original/train/AD'
data_dir = 'H:/final_project/PatternAnalysis_2023_Shan_Jiang/recognition/SuperResolutionShanJiang/AD_NC/train/AD'


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




