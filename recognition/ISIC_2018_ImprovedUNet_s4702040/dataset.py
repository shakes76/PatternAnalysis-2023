"""
This file implements a custom data set to be used in train.py for loading the data

The class loads images and masks together and applies the same 
transform for each mask/image pair is there is one available

NOTES:
- Converts the images to RGB values and the masks to greyscale values
- Does not require normalisation as the ranges for the RGB/greyscale values are already [0,1]
- Images/masks are only loaded if they have the prefix "ISIC" in their file name
"""

import torch
from PIL import Image
import os

# Uses as reference: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        prefix = "ISIC" # define the starting name of files to be loaded
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.images = [image for image in self.images if image.startswith(prefix)] # only get image files with prefix
        self.masks = os.listdir(mask_dir)
        self.masks = [mask for mask in self.masks if mask.startswith(prefix)] # only get mask files with prefix

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB') # converts image to RGB values
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        mask = Image.open(mask_path).convert('L') # converts mask to greyscale values
        if self.transform:
            # Apply transforms to image/mask if available
            image = self.transform(image)
            mask = self.transform(mask)
        return (image, mask)
