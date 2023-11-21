"""
This file is used to load the dataset and create the dataloader.
"""
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import numpy as np
from utils import *

# For the purpose of this report, we want to downscale the images by a factor of 4, before 
# upscaling by a factor of 4. This is to simulate the effect of a low resolution image.


def load_train_data(path: str):
    """
    Function that loads the training data. Returns train dataloader.
    """
    # TRANSFORMS
    training_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.ImageFolder(root=path, transform=training_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    if os.path.exists("figures/training_images.png"):
        print("Overwriting existing training image picture!")
    else:
        print("Creating training image picture!")

    # Plot some training images
    real_batch = next(iter(train_loader))
    num_images = 32
    plt.figure(figsize=(16, 16)) 
    plt.axis("off")
    plt.title("Training Images", fontsize=24)
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                real_batch[0][:num_images], nrow=4, padding=2, normalize=True 
            ).cpu(),
            (1, 2, 0),
        )
    )
    plt.savefig("figures/unprocessed_images.png")

    print("Training image picture created!")
    return train_loader

def load_test_data(path: str):
    """
    Function that loads the test daMdReturns test dataloader.
    """
    # TRANSFORMS
    test_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    test_dataset = torchvision.datasets.ImageFolder(root=path, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return test_loader