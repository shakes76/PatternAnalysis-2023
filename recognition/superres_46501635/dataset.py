import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import math
import torch.nn as nn

class ADNIDataset(Dataset):
    def __init__(self, root_dir, downsample_factor=4, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            downsample_factor (int): Factor by which the image will be downsampled.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.downsample_factor = downsample_factor
        self.image_files = []
        self.labels = []

        for subfolder, label in [('AD', 0), ('NC', 1)]:
            files_in_subfolder = [os.path.join(dp, f) for dp, dn, filenames in os.walk(os.path.join(root_dir, subfolder)) for f in filenames if f.endswith('.jpeg')]
            self.image_files.extend(files_in_subfolder)
            self.labels.extend([label] * len(files_in_subfolder))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name)

        # Downsample the image
        downsampled_image = image.resize((image.width // self.downsample_factor, image.height // self.downsample_factor), Image.BICUBIC)

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
            downsampled_image = self.transform(downsampled_image)

        label = self.labels[idx]

        return downsampled_image, image, label

def image_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Assuming Min-Max normalization to [0,1]. Adjust if using Z-score normalization.
    ])

def get_dataloaders(root_dir, batch_size=32):
    # Transforms
    transform = image_transform()

    # Datasets
    train_dataset = ADNIDataset(os.path.join(root_dir, 'train'), transform=transform)
    test_dataset = ADNIDataset(os.path.join(root_dir, 'test'), transform=transform)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader