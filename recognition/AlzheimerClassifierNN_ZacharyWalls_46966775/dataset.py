# For training and testing the ADNI dataset for Alzheimer's disease was utilised which can be found here; https://adni.loni.usc.edu/
# Go to DOWNLOAD -> ImageCollections -> Advanced Search area to download the data
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from collections import defaultdict
from PIL import Image
import torch.utils.data as data
import numpy as np

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
                uuid = "_".join(filename.split("_")[:-1])
                patient_data[uuid].append(os.path.join(self.root_dir, filename))

        # Sort the data based on the image number
        for uuid, slices in patient_data.items():
            slices.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

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
        tensors = torch.stack(slices, dim=1)  # (channels, depth, height, width)
        label = torch.tensor(self.label, dtype=torch.long)
        return tensors, label


def get_data_loaders():
    print("Initializing data transformations for dataset loading...")
    # Now use the computed mean and std for normalization in transformations
    train_transform = transforms.Compose(
        [
            transforms.CenterCrop(_size),
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
        train_dataset, batch_size=4, shuffle=True, num_workers=2
    )
    print("Training loader ready.\n")
    print("Loading and grouping testing data by UUID...")
    # Load the full test dataset
    test_dataset_AD = BrainScan3DDataset(
        root_dir="data/test/AD", transform=test_transform, label=0
    )
    test_dataset_NC = BrainScan3DDataset(
        root_dir="data/test/NC", transform=test_transform, label=1
    )
    test_dataset = torch.utils.data.ConcatDataset([test_dataset_AD, test_dataset_NC])

    # Calculate the desired subset size (1/4 of the test data)
    subset_size = len(test_dataset) // 8

    # Shuffle the dataset and select a subset
    indices = list(range(len(test_dataset)))
    np.random.shuffle(indices)
    subset_indices = indices[:subset_size]
    subset_sampler = data.SubsetRandomSampler(subset_indices)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, sampler=subset_sampler, num_workers=2
    )

    print(f"Testing data loaded with {subset_size} samples (1/8 subset).")
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
