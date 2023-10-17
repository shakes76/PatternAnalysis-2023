"""
File for all utilities/constants used in the model/other files.
"""
import os
import torch

# CONSTANTS
# HYPERPARAMETERS
upscale_factor = 4
num_epochs = 100
learning_rate = 0.001
channels = 1

image_width = 256
image_height = 240
downscale_factor = 4
new_width = image_width // downscale_factor
new_height = image_height // downscale_factor
upscale_factor = 4
batch_size = 8

ngpu = torch.cuda.device_count() # number of GPUs available. Use 0 for CPU mode.
num_workers = 2 * ngpu if ngpu > 1 else 2 # number of subprocesses to use for data loading
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# PATHS FOR LOCAL DEVELOPMENT
directory = os.path.abspath('./data/AD_NC')
train_path = os.path.join(directory, 'train')