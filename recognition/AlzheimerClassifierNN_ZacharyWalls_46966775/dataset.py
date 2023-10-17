# For training and testing the ADNI dataset for Alzheimer's disease was utilised which can be found here; https://adni.loni.usc.edu/
# Go to DOWNLOAD -> ImageCollections -> Advanced Search area to download the data

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os
from collections import defaultdict
from PIL import Image

# Same size utilized from Google's paper on ViT
# Images are converted to this size x size
_size = 224


def get_data_loaders():
    print("Initializing data transformations for dataset loading...")

    # Now use the computed mean and std for normalization in transformations
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(_size),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.14147302508354187, std=0.2420143187046051),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(_size),
            transforms.CenterCrop(_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.14147302508354187, std=0.2420143187046051),
        ]
    )

    print("Loading training data...")
    train_dataset = datasets.ImageFolder(root="data/train", transform=train_transform)
    print(f"Training data loaded with {len(train_dataset)} samples.")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=1
    )
    print("Training loader ready.\n")

    print("Loading testing data...")
    test_dataset = datasets.ImageFolder(root="data/test", transform=test_transform)
    print(f"Testing data loaded with {len(test_dataset)} samples.")

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=1
    )
    print(f"Testing loader ready.\n")

    return train_loader, test_loader


# Please note within the development environment the data was loaded in the following structure
# data/
#    ├── train/
#    │   ├── AD/
#    │   └── NC/
#    └── test/
#        ├── AD/
#        └── NC/
