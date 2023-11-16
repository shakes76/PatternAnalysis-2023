# dataset.py

import torch
from torchvision import datasets, transforms

# Constants
BATCH_SIZE = 32
DATASET_PATH = './OASIS'
NUM_WORKERS = 1

"""
Data preprocessing
Define a series of transformations to preprocess the image data
"""
preproc_transform = transforms.Compose([
    # Convert images to grayscale (1 channel)
    transforms.Grayscale(num_output_channels=1),
    # Convert images to tensors
    transforms.ToTensor(),
    # Normalize pixel values to the range [-1, 1] for grayscale
    transforms.Normalize((0.5,), (0.5,)) 
])

""" 
Data loaders
Create data loaders for the training, validation, and test datasets:
"""

# Training DataLoader
train_loader = torch.utils.data.DataLoader(
    # Use the datasets.ImageFolder class to load images from the 'train_images' directory
    datasets.ImageFolder(DATASET_PATH + '/train_images', transform=preproc_transform),
    batch_size=BATCH_SIZE, # Batch size for training
    shuffle=True, # Shuffle the data for training
    num_workers=NUM_WORKERS # Number of worker processes for data loading
)

# Validation DataLoader
val_loader = torch.utils.data.DataLoader(
    # Use the datasets.ImageFolder class to load images from the 'validat_images' directory
    datasets.ImageFolder(DATASET_PATH + '/validat_images', transform=preproc_transform),
    batch_size=BATCH_SIZE, # Batch size for validation
    shuffle=False, #  Do not shuffle the data for validation
    num_workers=NUM_WORKERS # Number of worker processes for data loading
)

# Test DataLoader
test_loader = torch.utils.data.DataLoader(
    # Use the datasets.ImageFolder class to load images from the 'test_images' directory
    datasets.ImageFolder(DATASET_PATH + '/test_images', transform=preproc_transform),
    batch_size=BATCH_SIZE, # Batch size for testing
    shuffle=False, # Do not shuffle the data for testing
    num_workers=NUM_WORKERS # Number of worker processes for data loading
)