from torchvision.datasets import ImageFolder
from torchvision.datasets import ImageFolder
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import random

def get_datasets(data_dir, transform=None):
    train_dataset = ImageFolder(root=data_dir + '/train', transform=transform)
    test_dataset = ImageFolder(root=data_dir + '/test', transform=transform)
    return train_dataset, test_dataset