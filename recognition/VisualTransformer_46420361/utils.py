"""Utility file for miscellaneous helper functions"""
import random

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


def plot_samples(train_dataset):
    num_rows = 5
    num_cols = num_rows

    # Create a figure with subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))

    # Iterate over the subplots and display random images from the training dataset
    for i in range(num_rows):
        for j in range(num_cols):
            # Choose a random index from the training dataset
            image_index = random.randrange(len(train_dataset))

            # Display the image in the subplot
            axs[i, j].imshow(train_dataset[image_index][0].permute((1, 2, 0)))

            # Set the title of the subplot as the corresponding class name
            axs[i, j].set_title(train_dataset.classes[train_dataset[image_index][1]], color="white")

            # Disable the axis for better visualization
            axs[i, j].axis(False)

    # Set the super title of the figure
    fig.suptitle(f"Random {num_rows * num_cols} images from the training dataset", fontsize=16, color="white")

    # Set the background color of the figure as black
    fig.set_facecolor(color='black')

    # Display the plot
    plt.show()
    
def calc_std_and_mean(root, batch_size, train=True):
    if train:
        path = root + 'test'
    else:
        path = root + 'test'
    
    # Add the 'transforms.ToTensor()' transformation to convert PIL images to tensors
    transform = transforms.Compose([transforms.ToTensor()])
    
    dataset = ImageFolder(path, transform=transform)
    dataloader = DataLoader(dataset, batch_size)
    
    mean = 0.
    std = 0.
    
    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        
    mean /= len(dataloader.dataset)
    std /= len(dataloader.dataset)
    
    print(mean)
    print(std)

