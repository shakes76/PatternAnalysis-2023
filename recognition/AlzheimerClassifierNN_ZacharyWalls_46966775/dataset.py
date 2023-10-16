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


class BrainScanDataset(Dataset):
    def __init__(self, root_dir, transform=None, label=None):
        self.root_dir = root_dir
        self.transform = transform
        self.patient_data = self._load_data()
        self.uuids = list(self.patient_data.keys())
        self.label = label

    def _load_data(self):
        patient_data = defaultdict(list)
        for filename in os.listdir(self.root_dir):
            if filename.endswith(".jpeg"):
                # Extract the UUID to group slices by patient
                uuid = "_".join(filename.split("_")[:-1])
                patient_data[uuid].append(os.path.join(self.root_dir, filename))

        # Sort the data to ensure slices are in the correct order
        for uuid, slices in patient_data.items():
            slices.sort()
        return patient_data

    def __len__(self):
        return len(self.uuids)

    def __getitem__(self, idx):
        patient_uuid = self.uuids[idx]
        image_paths = self.patient_data[patient_uuid]

        # Loading and transforming each slice, then stacking along a new dimension
        tensors = torch.stack(
            [self.transform(Image.open(p)) for p in image_paths], dim=0
        )
        label = torch.tensor(self.label, dtype=torch.long)

        return tensors, label


# We use this function to generate a normalization value for the dataset
def compute_mean_std(loader):
    """
    Compute the mean and standard deviation of the dataset.
    """
    mean = 0.0
    squared_mean = 0.0
    std = 0.0
    for images, _ in loader:
        mean += images.mean()
        squared_mean += (images**2).mean()

    mean /= len(loader)
    squared_mean /= len(loader)
    std = (squared_mean - mean**2) ** 0.5
    return mean.item(), std.item()


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
            transforms.RandomRotation(degrees=8),
            transforms.RandomAdjustSharpness(sharpness_factor=0.8, p=0.3),
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

    print("Loading and grouping training data by UUID...")
    # Get the UUIDs from both datasets
    train_dataset_AD = BrainScanDataset(
        root_dir="data/train/AD", transform=train_transform, label=0
    )
    train_dataset_NC = BrainScanDataset(
        root_dir="data/train/NC", transform=train_transform, label=1
    )

    all_uuids = train_dataset_AD.uuids + train_dataset_NC.uuids
    torch.random.manual_seed(42)  # Set a seed for reproducibility
    shuffled_uuids = torch.randperm(len(all_uuids)).tolist()

    # Use the shuffled UUIDs for both datasets
    train_dataset_AD.uuids = [
        all_uuids[i] for i in shuffled_uuids if all_uuids[i] in train_dataset_AD.uuids
    ]
    train_dataset_NC.uuids = [
        all_uuids[i] for i in shuffled_uuids if all_uuids[i] in train_dataset_NC.uuids
    ]

    train_dataset = torch.utils.data.ConcatDataset([train_dataset_AD, train_dataset_NC])
    print(f"Training data loaded with {len(train_dataset)} samples.")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False, num_workers=4
    )
    print("Training loader ready.\n")

    print("Loading and grouping testing data by UUID...")
    test_dataset_AD = BrainScanDataset(
        root_dir="data/test/AD", transform=test_transform, label=0
    )
    test_dataset_NC = BrainScanDataset(
        root_dir="data/test/NC", transform=test_transform, label=1
    )
    test_dataset = torch.utils.data.ConcatDataset([test_dataset_AD, test_dataset_NC])
    print(f"Testing data loaded with {len(test_dataset)} samples.")

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4
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
