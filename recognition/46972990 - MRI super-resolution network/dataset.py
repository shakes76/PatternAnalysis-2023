"""
Filename: dataset.py
Author: Benjamin Guy
Date: 15/10/2023
Description: This file loads the data from the ADNI dataset and preprocesses it into data loaders.
"""

import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split, Dataset

# Paths to the dataset
BASE_PATH = "C:\\Users\\User\\OneDrive\\Bachelor of Computer Science\\Semester 6 2023\\COMP3710\\ADNI_AD_NC_2D\\AD_NC"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH = os.path.join(BASE_PATH, "test")

# Image size and batch size
IMG_WIDTH = 256
IMG_HEIGHT = 240
BATCH_SIZE = 32

class DownsampleTransform:
    """
    This class will downsample an image by a default scale factor of 4.
    """
    def __init__(self, scale_factor=4):
        self.down_transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT // scale_factor, IMG_WIDTH // scale_factor))
        ])
    
    def __call__(self, img):
        return self.down_transform(img)
    
class SuperResolutionDataset(Dataset):
    """
    This class will create datasets consisting of downsampled images and original images.
    """
    def __init__(self, path):
        # Convert images to grayscale and resize
        self.resize_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH))
        ])
        self.data = datasets.ImageFolder(root=path, transform=self.resize_transform)
        
        # Transform images and normalize
        self.down_transform = transforms.Compose([
            DownsampleTransform(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.orig_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        orig_img, _ = self.data[idx]
        return self.down_transform(orig_img), self.orig_transform(orig_img)

def get_train_and_validation_loaders():
    # Load dataset and get sizes
    full_dataset = SuperResolutionDataset(TRAIN_PATH)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size) # 80% train, 20% validation
    val_size = total_size - train_size

    # Set seed for random split
    torch.manual_seed(1)
    
    # Create data loaders for train and validation datasets
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, validation_loader

def get_test_loader():
    # Load dataset and get the test data loader
    dataset = SuperResolutionDataset(TEST_PATH)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return data_loader