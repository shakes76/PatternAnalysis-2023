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
_size = 240


class BrainScan3DDataset(Dataset):
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

        # Loading and transforming each slice
        slices = [
            self.transform(Image.open(p).convert("L")) for p in image_paths
        ]  # Convert to grayscale

        # Stacking along the depth dimension
        tensors = torch.stack(slices, dim=2)  # (channels, height, depth, width)

        label = torch.tensor(self.label, dtype=torch.long)

        return tensors, label


def get_data_loaders():
    print("Initializing data transformations for dataset loading...")

    # Now use the computed mean and std for normalization in transformations
    train_transform = transforms.Compose(
        [
            transforms.Resize(_size),
            transforms.CenterCrop(_size),
            transforms.RandomRotation(degrees=3),
            transforms.RandomAdjustSharpness(sharpness_factor=0.8, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.14147302508354187, std=0.2420143187046051),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(_size),
            transforms.CenterCrop(_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.14147302508354187, std=0.2420143187046051),
        ]
    )

    print("Loading and grouping training data by UUID...")
    train_dataset_AD = BrainScan3DDataset(
        root_dir="data/train/AD", transform=train_transform, label=0
    )
    train_dataset_NC = BrainScan3DDataset(
        root_dir="data/train/NC", transform=train_transform, label=1
    )

    train_dataset = torch.utils.data.ConcatDataset([train_dataset_AD, train_dataset_NC])
    print(f"Training data loaded with {len(train_dataset)} samples.")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=6
    )
    print("Training loader ready.")

    print("Loading and grouping testing data by UUID...")
    test_dataset_AD = BrainScan3DDataset(
        root_dir="data/test/AD", transform=test_transform, label=0
    )
    test_dataset_NC = BrainScan3DDataset(
        root_dir="data/test/NC", transform=test_transform, label=1
    )
    test_dataset = torch.utils.data.ConcatDataset([test_dataset_AD, test_dataset_NC])
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
