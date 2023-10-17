"""
Source code for training, validation, testing, and saving of the model
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from dataset import *
from modules import *

# Load in the data
train_data = prep_data(DATA_PATH + 'train', IMG_DIM, BATCH_SIZE, SHIFT)

# Calculate training variance across pixels
full_train_data = train_data.unbatch()
num_train_images = full_train_data.reduce(np.int32(0), lambda x, _: x + 1).numpy()
num_train_pixels = num_train_images * IMG_DIM ** 2

image_sum = full_train_data.reduce(np.float32(0), lambda x, y: x + y).numpy().flatten()
tot_pixel_sum = image_sum.sum()
mean_pixel_train = tot_pixel_sum / num_train_pixels

# Calculate training variance
image_sse = full_train_data.reduce(np.float32(0), lambda x, y: x + (y - mean_pixel_train) ** 2).numpy().flatten()
var_pixel_train = image_sse.sum() / (num_train_pixels - 1)

# Have to go to work :( - tentative plane:

"""
1. Create VQ-VAE MODEL
2. Compile + Fit 
3. Save
4. Plot
5. Test / generate (in predict.py)

"""