"""
File: dataset.py
Author: Arshia Sharma 
Description: Creates custom ISIC dataset for Improved Unet model. 

Dependencies: PIL numpy torch torchvision 
"""

import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# Global constants for the dimensions to which images and masks will be resized
TRANSFORMED_X = 256  # width after resizing
TRANSFORMED_Y= 256  # height after resizing

"""
    PyTorch Dataset class for loading ISIC melanoma detection dataset.

    Parameters:
    - img_path (str): Path to the directory containing image files.
    - mask_path (str): Path to the directory containing mask files.
    - transform (call): Transform to be applied on the images and masks.

    Methods:
    - __len__(): Returns the total number of images in the dataset.
    - __getitem__(idx): Returns the image and its corresponding mask at the given index `idx`. Resizes image and converts to correct colour channels. 
    - match_mask_to_image(img_filename): Converts image filename to its corresponding mask filename.
    """
class ISICDataset(Dataset):
    def __init__(self, img_path, mask_path, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        # filter out text files from file names
        self.img_filenames = sorted([f for f in os.listdir(img_path) if f not in ["ATTRIBUTION.txt", "LICENSE.txt"]])
        self.transform = transform

    def __len__(self):
        return len(self.img_filenames)

    def match_mask_to_image(self, img_filename):
        base_name = os.path.splitext(img_filename)[0]  # This removes the .jpg or any extension
        return base_name + '_segmentation.png'

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_path, self.img_filenames[idx])
        mask_name = os.path.join(self.mask_path, self.match_mask_to_image(self.img_filenames[idx]))

        img = Image.open(img_name).convert('RGB')  # Convert to RGB
        mask = Image.open(mask_name).convert('L')  # Convert to grayscale for segmentation masks.

        img = img.resize((TRANSFORMED_Y, TRANSFORMED_X))
        mask = mask.resize((TRANSFORMED_Y, TRANSFORMED_X), Image.NEAREST)

        if self.transform:
            img = self.transform(img)
            mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0  # Convert to single channel tensor

        return img, mask


"""
Helper method to compute mean and standard deviations for the train dataset.
Takes in an input loader.

NB: used for preprocessing, not used directly in train or predict. 

Parameters:
loader (DataLoader): A DataLoader object that provides batches of data.

Returns:
mean (tensor): The mean values for each channel in the dataset.
std (tensor): The standard deviations for each channel in the dataset.
"""

def compute_mean_std(loader):
    mean = 0.0
    squared_mean = 0.0
    total_samples = 0.0

    for images, _ in loader:
        # flatten image tensor and calculated mean, squared mean across all pixel values.

        images_flat = images.view(images.size(0), images.size(1), -1)
        mean += images_flat.mean(2).sum(0)
        squared_mean += (images_flat ** 2).mean(2).sum(0)
        total_samples += images.size(0)

    # calculate mean and std dev across whole dataset.
    mean /= total_samples
    squared_mean /= total_samples
    std = (squared_mean - mean**2)**0.5

    return mean, std
