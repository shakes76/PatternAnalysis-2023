"""
This file is used to load the dataset and create the dataloader.
"""
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from utils import *

# For the purpose of this report, we want to downscale the images by a factor of 4, before 
# upscaling by a factor of 4. This is to simulate the effect of a low resolution image.

def load_data():
    """
    Function that loads the data and creates the dataloader. Returns the dataloader.
    """
    print("Loading data...")

    # TRANSFORMS
    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.ImageFolder(root=train_path, transform=training_transform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if os.path.exists("training_images.png"):
        print("Overwriting existing training image picture!")
    else:
        print("Creating training image picture!")

    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(16, 16)) 
    plt.axis("off")
    plt.title("Training Images", fontsize=24)
    plt.imshow(
        np.transpose(
            vutils.make_grid(
            real_batch[0][0].to(device)[:16], nrow=4, padding=2, normalize=True 
            ).cpu(),
            (1, 2, 0),
        )
    )
    plt.savefig("training_images.png")

    print("Training image picture created!")
    print("Data loaded!")
    return dataloader