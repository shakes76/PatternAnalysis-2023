"""
Data loading and preprocessing
"""
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
from config import *

# Performs the image downsampling
def downsample_tensor(original: torch.Tensor):
    return transforms.Resize([
        original_height // dimension_reduce_factor, 
        original_width // dimension_reduce_factor
    ],antialias=True)(original)


# Gets the training data loader, with appropriate data transformations
def get_train_dataloader(shuffle=True):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    AD_dataset = ImageFolder(root=AD_train_dir, transform=transform)
    NC_dataset = ImageFolder(root=NC_train_dir, transform=transform)

    # use both AD and NC samples to train model
    dataset = ConcatDataset([AD_dataset, NC_dataset])

    # Create a data loader to iterate through the dataset
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


# Gets the test data loader, with no data transformations
def get_test_dataloader(shuffle=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    AD_dataset = ImageFolder(root=AD_test_dir, transform=transform)
    NC_dataset = ImageFolder(root=NC_test_dir, transform=transform)

    # use both AD and NC samples to train model
    dataset = ConcatDataset([AD_dataset, NC_dataset])

    # Create a data loader to iterate through the dataset
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


# Helper function to debug what the downsampled images look like, and to
# check their dimensions
def save_dimensions(data_loader):
    for original, _ in data_loader:
        print('original.shape', original.shape)

        # Look at the first image
        output = original[0]
        # Downsample to get input
        input = downsample_tensor(output)

        all_inputs = downsample_tensor(original)
        print('batch inputs shape:', all_inputs.shape)
        
        input_dims = input.shape
        output_dims = output.shape
        
        # Display the input image
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(input.permute(1, 2, 0))  # Convert tensor to numpy format (C, H, W) -> (H, W, C)
        plt.title(f"Input Image Dimensions: {input_dims}")
        plt.axis('off')  # Turn off axis labels
        
        # Display the first output image
        plt.subplot(1, 2, 2)
        plt.imshow(torch.clamp(output.permute(1, 2, 0), 0, 1))  # Convert tensor to numpy format (C, H, W) -> (H, W, C)
        plt.title(f"Output Image Dimensions: {output_dims}")
        plt.axis('off') # Turn off axis labels
        
        plt.savefig(image_dir + 'dimensions.png')
        # plt.show()
        plt.close()
        
        break  # Stop after the first batch to print/display only the first pair of images
