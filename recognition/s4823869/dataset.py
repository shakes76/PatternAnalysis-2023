import glob
import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

NORMALISE = 255.0
CENTRE = 0.5
IMAGE_DIM = 80
ENC_IN_SHAPE = (1, 80, 80)

GREYSCALE = "L"  # Grayscale mode in PyTorch

SPLITS = 3
BASE = "keras_png_slices_data/keras_png_slices_"
TRAINING = BASE + "train/*"
TESTING = BASE + "test/*"
VALIDATION = BASE + "validate/*"

def normalise(data):
    """
    Normalize the input data.

    data - Input data to be normalized

    Returns normalized data
    """
    # Normalize the data by dividing by the normalization factor
    data = data / NORMALISE
    return data

class CustomDataset(Dataset):
    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img = Image.open(self.data_paths[idx])
        if self.transform:
            img = self.transform(img)
        return img

def get_ttv(subset_size=None):
    """
    Read in the training/testing/validation datasets from local files.
    Mostly repurposed from the original demo.

    return    - the training, testing, and validation datasets
    """
    train_data = glob.glob(TRAINING)
    test_data = glob.glob(TESTING)
    val_data = glob.glob(VALIDATION)

    # Check if the directories exist and contain files
    if not train_data:
        print("No training data found. Please check file paths or obtain the missing data.")
        return None, None, None

    if not test_data:
        print("No testing data found. Using a subset of training data for testing.")
        test_data = train_data[:subset_size]

    if not val_data:
        print("No validation data found. Using a subset of training data for validation.")
        val_data = train_data[subset_size:2 * subset_size]

    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((IMAGE_DIM, IMAGE_DIM)),
                                    transforms.ToTensor()])

    train_dataset = CustomDataset(data_paths=train_data, transform=transform)
    test_dataset = CustomDataset(data_paths=test_data, transform=transform)
    val_dataset = CustomDataset(data_paths=val_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader, val_loader

def preview(dataset, n):
    """
    Show the first n^2 images of the dataset in an n x n grid

    dataset    - training / testing / validation dataset to preview
    n        - length of the preview square grid
    """
    fig, axes = plt.subplots(n, n, figsize=(8, 8))
    for i in range(n):
        for j in range(n):
            ind = (n * i) + j
            img = dataset.dataset[ind].numpy().squeeze()
            axes[i, j].imshow(img, cmap=GREYSCALE)
            axes[i, j].axis('off')

    plt.show()
