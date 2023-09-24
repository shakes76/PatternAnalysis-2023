"""
Data loading and preprocessing
"""
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
from config import *


def downsample_tensor(original: torch.Tensor):
    return transforms.Resize([
        original_height // dimension_reduce_factor, 
        original_width // dimension_reduce_factor
    ],antialias=True)(original)


def get_dataloader():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    AD_dataset = ImageFolder(root=AD_dir, transform=transform)
    NC_dataset = ImageFolder(root=NC_dir, transform=transform)

    # use both AD and NC samples to train model
    dataset = ConcatDataset([AD_dataset, NC_dataset])

    # Create a data loader to iterate through the dataset
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)