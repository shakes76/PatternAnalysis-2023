"""
datasplit.py

creates the train loader and test loader.

Author: Atharva Gupta
Date Created: 17-10-2023
"""
import torch
from torchvision import datasets, transforms

def create_transforms(crop_size=240):
    """
    Create a transformation pipeline with customization options.
    Args:
        crop_size (int): Size for random cropping.

    Returns:
        A data transformation pipeline.
    """
    transformation = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.RandomCrop(crop_size),  # Randomly crop the image
        transforms.ToTensor(),  # Convert to a PyTorch tensor
        transforms.Lambda(lambda x: x[0][:, :, None])  # Convert to 240x240x1 format
    ])
    return transformation

def create_data_loader(dir, batch_size=32, shuffle=True, num_workers=0, crop_size=240):
    """
    Create a data loader for the specified directory.
    Args:
        dir (str): Path to the dataset directory.
        batch_size (int): Batch size for the data loader.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of data loading workers.
        crop_size (int): Size for random cropping.

    Returns:
        A PyTorch data loader.
    """
    data_transform = create_transforms(crop_size=crop_size)
    dataset = datasets.ImageFolder(dir, transform=data_transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader

# Example usage:
# Create data loaders for training and testing
train_loader = create_data_loader(dir='C:/Users/hp/Desktop/comp3710/PatternAnalysis-2023/recognition/s48195609/AD_NC/train',
                                  batch_size=32, shuffle=True, num_workers=4)
test_loader = create_data_loader(dir='C:/Users/hp/Desktop/comp3710/PatternAnalysis-2023/recognition/s48195609/AD_NC/test',
                                 batch_size=32, shuffle=False, num_workers=4)
