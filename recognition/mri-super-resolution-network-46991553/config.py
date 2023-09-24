"""
Hyperparameters and other meaningful constants
"""

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