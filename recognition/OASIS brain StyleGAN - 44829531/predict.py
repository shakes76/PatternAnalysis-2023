"""
predict.py

Author: Ethan Jones
Student ID: 44829531
COMP3710 OASIS brain StyleGAN project
Semester 2, 2023
"""

import os

# Hyparparameters
EPOCHS = 5
BATCH_SIZE = 32

# Inputs
INPUT_IMAGES_PATH = "C:\\Users\\ethan\\Desktop\\COMP3710" \
                    "\\keras_png_slices_train "
# INPUT_IMAGES_PATH = "/home/groups/comp3710/OASIS/keras_png_slices_train"

# Results
RESULT_IMAGE_PATH = "figures"
RESULT_WEIGHT_PATH = "figures"
RESULT_IMAGE_COUNT = 3
PLOT_LOSS = True

