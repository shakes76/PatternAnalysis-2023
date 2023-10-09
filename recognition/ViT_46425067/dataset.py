"""
Created on Wednesday September 20 2023

Script to load in the ADNI dataset as a pytorch DataLoader

@author: Rodger Xiang s4642506
"""

import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import platform

# Paths to Data depending on OS system
OS = platform.system()
if OS == "Windows":
    TRAIN_DATA_PATH = Path("E:/UNI 2023 SEM 2/COMP3710/Lab3/recognition/ViT_46425067/AD_NC/train")
    TEST_DATA_PATH = Path("E:/UNI 2023 SEM 2/COMP3710/Lab3/recognition/ViT_46425067/AD_NC/test")
else:
    TRAIN_DATA_PATH = Path("./AD_NC/train/")
    TEST_DATA_PATH = Path("./AD_NC/test/")


def load_data(batch_size, image_size):
    """
    Loads data from folders using the TRAIN_DATA_PATH and TEST_DATA_PATH and
    
    Args:
        batch_size (int): the size of each batch
        image_size (int): value to resize image to.

    Returns:
        DataLoader: train and test dataloaders
    """
    # Training Transformations
    train_transforms = transforms.Compose([
        transforms.Resize((image_size,image_size)), 
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.Normalize(mean=(0.1156), std=(0.2198), inplace=True),
    ])
    
    # test data transformations
    test_transforms = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize(mean=(0.1156), std=(0.2198), inplace=True),
    ])

    # create datasets for train and test data
    train_dataset = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=TEST_DATA_PATH, transform=test_transforms)

    # create DataLoaders for train and test datasets
    train_loader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    return train_loader, test_loader