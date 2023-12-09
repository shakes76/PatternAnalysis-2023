"""
dataset.py

This module provides a utility for loading the ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset.
ADNI dataset is used in the field of neuroimaging to study Alzheimer's disease.

Features:
- ADNIDataset: A custom PyTorch Dataset class to facilitate loading of the AD and NC images.
- get_data_loader: A utility function to easily retrieve a DataLoader instance for batching.

Usage:
    dataloader = get_data_loader("/path/to/ADNI_data", train=True, transform=your_transforms, batch_size=64)

Structure:
The expected directory structure of the ADNI dataset is:
    /path/to/ADNI_data/
    ├── train/
    │   ├── AD/
    │   │   ├── image1.png
    │   │   ├── image2.png
    │   │   └── ...
    │   └── NC/
    │       ├── image1.png
    │       ├── image2.png
    │       └── ...
    └── test/
        ├── AD/
        │   ├── image1.png
        │   ├── image2.png
        │   └── ...
        └── NC/
            ├── image1.png
            ├── image2.png
            └── ...

Classes:
    - AD (Alzheimer's Disease)
    - NC (Normal Control)

Author:
    Santiago Rodrigues (46423232)
"""


import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ADNIDataset(Dataset):
    """
    Dataset class for the ADNI dataset, which includes images from two classes: AD (Alzheimer's Disease) and NC (Normal Control).
    """
    
    def __init__(self, root_dir, train=True, transform=None):
        """
        Initialize the dataset.

        Parameters:
        - root_dir (str): Root directory where the 'AD_NC' dataset resides.
        - train (bool, optional): Whether to load the training data. If False, it will load testing data. Defaults to True.
        - transform (callable, optional): Optional transform to be applied on the images. Defaults to None.
        """

        # Root directory path
        self.root_dir = root_dir
        self.transform = transform
        
        # Depending on the choice of training/testing, set the data directory accordingly
        self.data_dir = os.path.join(root_dir, "train" if train else "test")
        
        # List all the AD and NC image file paths
        self.ad_files = [os.path.join(self.data_dir, "AD", fname) for fname in os.listdir(os.path.join(self.data_dir, "AD"))]
        self.nc_files = [os.path.join(self.data_dir, "NC", fname) for fname in os.listdir(os.path.join(self.data_dir, "NC"))]
        
        # Merge both the lists
        self.files = self.ad_files + self.nc_files

    def __len__(self):
        """Returns the total number of images."""
        return len(self.files)
    
    def __getitem__(self, idx):
        """
        Retrieve the image at the given index.

        Parameters:
        - idx (int): Index of the image to retrieve.

        Returns:
        - img (PIL.Image): Image at the specified index.
        """

        # Load the image from its path and convert it to grayscale
        img_path = self.files[idx]
        img = Image.open(img_path).convert('L')
        
        # If any transformations are specified, apply them
        if self.transform:
            img = self.transform(img)

        return img

def get_data_loader(root_dir, train=True, transform=None, batch_size=32, shuffle=True):
    """
    Utility function to get a DataLoader for the ADNI dataset.
    """

    # Create an instance of the dataset
    dataset = ADNIDataset(root_dir, train, transform)
    
    # Create a DataLoader instance from the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader
