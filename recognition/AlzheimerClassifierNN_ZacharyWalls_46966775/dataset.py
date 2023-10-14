# For training and testing the ADNI dataset for Alzheimer's disease was utilised which can be found here; https://adni.loni.usc.edu/
# Go to DOWNLOAD -> ImageCollections -> Advanced Search area to download the data

import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split

# Same size utilized from Google's paper on ViT
_size = 240

# We use this function to generate a normalization value for the dataset
def compute_mean_std(loader):
    """
    Compute the mean and standard deviation of the dataset.
    """
    mean = 0.0
    squared_mean = 0.0
    std = 0.0
    for images, _ in loader:
        mean += images.mean([0, 2, 3])
        squared_mean += (images**2).mean([0, 2, 3])

    mean /= len(loader)
    squared_mean /= len(loader)
    std = (squared_mean - mean**2) ** 0.5
    return mean.tolist(), std.tolist()


def get_data_loaders():
    print("Initializing data transformations for dataset loading...")
    basic_transform = transforms.Compose(
        [transforms.CenterCrop((224, 224)), transforms.ToTensor()]
    )

    print("Loading training data for normalization...")
    temp_train_dataset = datasets.ImageFolder(
        root="data/train", transform=basic_transform
    )
    temp_train_loader = torch.utils.data.DataLoader(
        temp_train_dataset, batch_size=32, shuffle=False
    )

    # Compute mean and std
    mean, std = compute_mean_std(temp_train_loader)
    print(f"Computed Mean: {mean}, Standard Deviation: {std}\n")

    # Now use the computed mean and std for normalization in transformations
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(_size),
            transforms.CenterCrop(_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # The rest of your function remains the same...
    print("Loading training data...")
    train_dataset = datasets.ImageFolder(root="data/train", transform=train_transform)
    print(f"Training data loaded with {len(train_dataset)} samples.")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=6
    )
    print("Training loader ready.")


    print("Loading testing data...")
    test_dataset = datasets.ImageFolder(root="data/test", transform=test_transform)
    print(f"Testing data loaded with {len(test_dataset)} samples.")

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=6
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
