# Import necessary libraries
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


# Define a function to create a data loader
def get_loader(dataset, log_resolution, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Define a series of data transformations to be applied to the images
    transform = transforms.Compose(
        [
            # Resize images to a specified resolution (2^log_resolution x 2^log_resolution)
            transforms.Resize((2 ** log_resolution, 2 ** log_resolution)),
            
            # Convert the images to PyTorch tensors
            transforms.ToTensor(),
            
            # Apply random horizontal flips to augment the data (50% probability)
            transforms.RandomHorizontalFlip(p=0.5),
            
            # Normalize the pixel values of the images to have a mean and standard deviation of 0.5
            transforms.Normalize(
                [0.5, 0.5, 0.5],  # Mean for each channel
                [0.5, 0.5, 0.5],  # Standard deviation for each channel
            ),
        ]
    )
    
    # Create an ImageFolder dataset object that loads images from the specified directory
    dataset = datasets.ImageFolder(root=dataset, transform=transform)
    
    # Create a data loader that batches and shuffles the data
    loader = DataLoader(
        dataset,
        batch_size=batch_size,  # Number of samples per batch
        shuffle=True,          # Shuffle the data for randomness
    )
    
    real_batch = next(iter(loader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    # Return the created data loader
    return loader

# Define constants and hyperparameters
DATASET                 = "./OASIS"  # Path to the dataset  # Use GPU if available, otherwise use CPU
EPOCHS                  = 301 # Number of training epochs
LEARNING_RATE           = 1e-3  # Learning rate for optimization
BATCH_SIZE              = 32  # Batch size for training
LOG_RESOLUTION          = 7  # Logarithmic resolution used for 128*128 images
Z_DIM                   = 256  # Dimension of the latent space
W_DIM                   = 256  # Dimension of the mapping network output
LAMBDA_GP               = 10  # Weight for the gradient penalty term

get_loader(DATASET, LOG_RESOLUTION, BATCH_SIZE)