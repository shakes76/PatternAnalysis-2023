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


def extract_patient_id(image_path):
    """Extract patient ID from image path."""
    base_name = os.path.splitext(os.path.basename(image_path))[0] # # Extract the base filename without the extension
    return base_name.split('_')[0] # Split by underscore and return the patient ID 

## Extract unique patient IDs
all_patient_ids_AD = list(set(extract_patient_id(path) for path in all_train_images_paths_AD))
all_patient_ids_NC = list(set(extract_patient_id(path) for path in all_train_images_paths_NC))

# Split patient IDs into training and validation sets (20%, 80% split)
train_patient_ids_AD, val_patient_ids_AD = train_test_split(all_patient_ids_AD, test_size=0.2, random_state=42) 
train_patient_ids_NC, val_patient_ids_NC = train_test_split(all_patient_ids_NC, test_size=0.2, random_state=42)

# Map patient IDs back to image paths
train_images_AD = [path for path in all_train_images_paths_AD if extract_patient_id(path) in train_patient_ids_AD]
val_images_AD = [path for path in all_train_images_paths_AD if extract_patient_id(path) in val_patient_ids_AD]
train_images_NC = [path for path in all_train_images_paths_NC if extract_patient_id(path) in train_patient_ids_NC]
val_images_NC = [path for path in all_train_images_paths_NC if extract_patient_id(path) in val_patient_ids_NC]




