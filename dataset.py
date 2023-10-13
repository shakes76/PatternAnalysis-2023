import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split


def get_image_paths_from_directory(directory_path, valid_extensions=[".jpg", ".jpeg", ".png"]):
    """Returns list of all image paths with valid extensions in the provided directory."""
    if not os.path.exists(directory_path):
        raise ValueError(f"The provided directory {directory_path} does not exist.")
    
    all_images = []
    for image_file in os.listdir(directory_path):
        if any(image_file.endswith(ext) for ext in valid_extensions):
            image_path = os.path.join(directory_path, image_file)
            all_images.append(image_path)
    return all_images

train_images_paths_NC = '/Users/noammendelson/Documents/REPORT-COMP3710/AD_NC/train/AD'
train_images_paths_AD = '/Users/noammendelson/Documents/REPORT-COMP3710/AD_NC/train/NC'

all_train_images_paths_NC = get_image_paths_from_directory(train_images_paths_NC)
all_train_images_paths_AD = get_image_paths_from_directory(train_images_paths_AD)


