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

# For the purpose of this report, we want to downscale the images by a factor of 4, before 
# upscaling by a factor of 4. This is to simulate the effect of a low resolution image.

# CONSTANTS
image_width = 256
image_height = 240
downscale_factor = 4
new_width = image_width // downscale_factor
new_height = image_height // downscale_factor
upscale_factor = 4
batch_size = 8


ngpu = torch.cuda.device_count() # number of GPUs available. Use 0 for CPU mode.
num_workers = 2 * ngpu if ngpu > 1 else 2 # number of subprocesses to use for data loading

# PATHS FOR LOCAL DEVELOPMENT
directory = os.path.abspath('./data/AD_NC')
train_path = os.path.join(directory, 'train')
# test_path = os.path.join(directory, 'test')

def load_data():
    """
    Function that loads the data and creates the dataloader. Returns the dataloader.
    """
    print("Loading data...")

    # TRANSFORMS
    transform_low_res = transforms.Compose([
        transforms.Resize((new_height, new_width)),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    transform_high_res = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    dataset_low_res = torchvision.datasets.ImageFolder(train_path, transform=transform_low_res)
    dataset_high_res = torchvision.datasets.ImageFolder(train_path, transform=transform_high_res)

    dataloader = torch.utils.data.DataLoader(
        [(low, high) for low, high in zip(dataset_low_res, dataset_high_res)],
        batch_size=batch_size,
        num_workers=num_workers,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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