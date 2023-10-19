"""
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
    VAL_DATA_PATH = Path("E:/UNI 2023 SEM 2/COMP3710/Lab3/recognition/ViT_46425067/AD_NC/validate")
    TEST_DATA_PATH = Path("E:/UNI 2023 SEM 2/COMP3710/Lab3/recognition/ViT_46425067/AD_NC/test")
else:
    TRAIN_DATA_PATH = Path("./AD_NC/train/")
    VAL_DATA_PATH = Path("./AD_NC/validate/")
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
    
    # validate, test data transformations
    val_transforms = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize(mean=(0.1156), std=(0.2198), inplace=True),
    ])
    
    # create data loaders for the data 
    train_loader = data_loader(TRAIN_DATA_PATH, train_transforms, batch_size, True)
    val_loader = data_loader(VAL_DATA_PATH, val_transforms, batch_size)
    test_loader = data_loader(TEST_DATA_PATH, val_transforms, batch_size)
    
    return train_loader, val_loader, test_loader

def data_loader(path, transform, batch_size, shuffle=False):
    """creates a pytorch DataLoader for the data

    Args:
        path (Path): absolute or relative path to data
        transform (torchvision.transforms): transforms applied to data
        batch_size (int): size of each batch
        shuffle (bool, optional): shuffle data. Default: False

    Returns:
        DataLoader : pytorch DataLoader of the data
    """
    dataset = datasets.ImageFolder(root=path, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader