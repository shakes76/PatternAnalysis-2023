"""
dataset.py: Custom dataset definitions for Siamese network training and classification.

Author: Rachit Chaurasia (s4823870)
Date: 20/10/2023

This module defines custom datasets for Siamese network training and image classification.
The SiameseDataset is designed for Siamese network training and classification. It provides
support for loading pairs of images and their corresponding labels, which indicate whether
the pairs are similar or dissimilar. The load_siamese_data function loads data for Siamese
network training, while the load_classify_data function loads data for image classification.

"""

import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from torchvision import transforms

# Setting the global path for dataset.
GLOBAL_PATH = 'C:\\Project 01 PYTORCH\\ADNI_AD_NC_2D'
AD_PATH = os.path.join(GLOBAL_PATH, 'AD_NC', 'train', 'AD')
CN_PATH = os.path.join(GLOBAL_PATH, 'AD_NC', 'train', 'NC')
AD_TEST_PATH = os.path.join(GLOBAL_PATH, 'AD_NC', 'test', 'AD')
CN_TEST_PATH = os.path.join(GLOBAL_PATH, 'AD_NC', 'test', 'NC')

# custom dataset for Siamese networks.
class SiameseDataset(Dataset):
    """
    Custom dataset for Siamese network training and classification.

    Args:
        path1 (list): List of file paths for the first set of images.
        path2 (list): List of file paths for the second set of images.
        labels (array-like): List of labels (0 or 1) indicating if the pairs are similar or dissimilar.
        transform (callable, optional): Optional data transformations to apply to the images.
    """
    def __init__(self, path1, path2, labels, transform=None):
        self.path1 = path1
        self.path2 = path2
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.path1)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            img1 (PIL.Image): First image.
            img2 (PIL.Image): Second image.
            label (int): Label (0 for dissimilar, 1 for similar).
        """
        img1 = Image.open(self.path1[idx]).convert('L')
        img2 = Image.open(self.path2[idx]).convert('L')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = self.labels[idx]

        return img1, img2, label

# Function to load data for Siamese network training.
def load_siamese_data(batch_size=32):
    """
    Load data for Siamese network training.

    Args:
        batch_size (int, optional): Batch size for the data loader.

    Returns:
        dataloader (DataLoader): DataLoader for Siamese network training.
    """
    pair_base = [os.path.join(CN_PATH, path) for path in os.listdir(CN_PATH)][::2]
    pair_ad = [os.path.join(AD_PATH, path) for path in os.listdir(AD_PATH)][:len(pair_base)]
    pair_cn = [os.path.join(CN_PATH, path) for path in os.listdir(CN_PATH)][1::4][:len(pair_base)]
    pair_compare = pair_cn + pair_ad
    labels = np.concatenate([np.zeros(len(pair_base)//2), np.ones(len(pair_base)//2)])

    transform = transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    siamese_dataset = SiameseDataset(pair_base, pair_compare, labels, transform=transform)
    dataloader = DataLoader(siamese_dataset, batch_size=batch_size, shuffle=True)

    return dataloader

# Function to load data for classification.
def load_classify_data(testing=False, batch_size=32):
    """
    Load data for classification.

    Args:
        testing (bool, optional): Set to True for loading test data, False for training data.
        batch_size (int, optional): Batch size for the data loader.

    Returns:
        dataloader (DataLoader): DataLoader for classification.
    """
    if not testing:
        ad_paths = [os.path.join(AD_PATH, path) for path in os.listdir(AD_PATH)]
        cn_paths = [os.path.join(CN_PATH, path) for path in os.listdir(CN_PATH)]
    else:
        ad_paths = [os.path.join(AD_TEST_PATH, path) for path in os.listdir(AD_TEST_PATH)]
        cn_paths = [os.path.join(CN_TEST_PATH, path) for path in os.listdir(CN_TEST_PATH)]

    paths = ad_paths + cn_paths
    labels = torch.cat((torch.ones(len(ad_paths)), torch.zeros(len(cn_paths))))

    transform = transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    classify_dataset = SiameseDataset(paths, paths, labels, transform=transform)
    dataloader = DataLoader(classify_dataset, batch_size=batch_size, shuffle=True)

    return dataloader
