"""
datasplit.py

Data loader for loading and preprocessing data.

Author: Atharva Gupta
Date Created: 17-10-2023
"""
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from parameter import DATA_LOAD_PATH, IMAGE_SIZE, BATCH_SIZE


def load_data():
    """
    Loads the dataset that will be used into PyTorch datasets.
    """
    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_LOAD_PATH, 'train'),
        transform=data_transforms
    )

    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_LOAD_PATH, 'test'),
        transform=data_transforms
    )

    # Split the test dataset into validation and test sets
    num_test_images = len(test_dataset)
    split_index = num_test_images // 2

    validation_dataset, test_dataset = torch.utils.data.random_split(
        test_dataset, [split_index, num_test_images - split_index]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, validation_loader, test_loader
