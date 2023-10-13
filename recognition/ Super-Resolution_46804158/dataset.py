import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms.functional as fn

import os
import pandas as pd
from torchvision.io import read_image


class CustomImageDataset():
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class ADNIDataset():
    def __init__(self):
        """
        Contains the data loader for loading and preprocessing the MRI data.
        """
        train_data_path = '/Users/mj/Documents/COMP3710_code/COMP3710_Report/AD_NC/train'
        test_data_path = '/Users/mj/Documents/COMP3710_code/COMP3710_Report/AD_NC/test'

        # Define data transformation an preprocess data
        transform_downscale = transforms.Compose([
            transforms.CenterCrop(240),
            transforms.Resize((60, 60)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        transform_target = transforms.Compose([
            transforms.CenterCrop(240),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Apply transformations to datasets
        train_dataset = ImageFolder(root=train_data_path, transform=transform_downscale)
        target_dataset =ImageFolder(root=train_data_path, transform=transform_target)
        test_dataset = ImageFolder(root=test_data_path, transform=transform_downscale)

        self.train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        self.target_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)


data = ADNIDataset()
train_img, train_label = next(iter(data.train_loader))
print(train_img.shape)
print(train_label)

test_img, test_label = next(iter(data.test_loader))
print(test_img.shape)
print(test_label)