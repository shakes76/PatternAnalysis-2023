"""
Hyperparameters and other meaningful constants
"""
import os

# Factor to reduce the width and height of the images by
dimension_reduce_factor = 2  # downsample factor of 4

# Original dimensions
original_width = 256
original_height = 240

# Directory to save images to
image_dir = "imgs/"

# Path to image folders - use both AD and NC when training
AD_dir = "data/AD_NC/train/AD-parent/"  # AD = Alzheimer’s disease
NC_dir = "data/AD_NC/train/NC-parent/"  # CN = Cognitive Normal

# Images per batch
batch_size = 128
# Number of epochs
num_epochs = 10

# Name to save the model as a file
model_filename = "super_resolution_model.pth"