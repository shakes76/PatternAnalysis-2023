"""
    File name: modules.py
    Author: Fanhao Zeng
    Date created: 11/10/2023
    Date last modified: 16/10/2023
    Python Version: 3.10.12
"""

import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ADNIDataset(Dataset):
    def __init__(self, data_path, classifier=False):
        super(ADNIDataset, self).__init__()

        self.classifier = classifier

        self.transform = transforms.ToTensor()

        # Load AD and NC images
        self.ad_path = os.path.join(data_path, 'AD')
        self.nc_path = os.path.join(data_path, 'NC')

        # Load images
        self.ad_images = [self.transform(Image.open(os.path.join(self.ad_path, img))) for img in
                          os.listdir(self.ad_path)]
        self.nc_images = [self.transform(Image.open(os.path.join(self.nc_path, img))) for img in
                          os.listdir(self.nc_path)]

        # Stack images into tensors
        self.ad_images = torch.stack(self.ad_images)
        self.nc_images = torch.stack(self.nc_images)

    def __len__(self):
        if not self.classifier:
            # Return the length of the smaller dataset
            return min(len(self.ad_images), len(self.nc_images))
        else:
            return 2 * min(len(self.ad_images), len(self.nc_images))

    def __getitem__(self, index):
        if not self.classifier:
            if index % 2 == 0:
                # Positive example (both images are AD)
                img1 = self.ad_images[index % len(self.ad_images)]  # Get the image at the current index
                img2 = self.ad_images[(index + 1) % len(self.ad_images)]  # Get the next image
                label = torch.tensor(1, dtype=torch.float)  # Set the label to 1
            else:
                # Negative example (one image is AD, the other is NC)
                img1 = self.ad_images[index % len(self.ad_images)]  # Get the image of ad at the current index
                img2 = self.nc_images[index % len(self.nc_images)]  # Get the image of nc at the current index
                label = torch.tensor(0, dtype=torch.float)  # Set the label to 0

            return img1, img2, label
        else:
            if index % 2 == 0:
                img = self.ad_images[index % len(self.ad_images)]
                label = torch.tensor(1, dtype=torch.float)
            else:
                img = self.nc_images[index % len(self.nc_images)]
                label = torch.tensor(0, dtype=torch.float)
            return img, label


def get_train_dataset(data_path):
    # Get the training dataset
    train_dataset = ADNIDataset(os.path.join(data_path, 'train'))
    return train_dataset


def get_test_dataset(data_path):
    # Get the test dataset
    test_dataset = ADNIDataset(os.path.join(data_path, 'test'))
    return test_dataset


def get_classifier_train_dataset(data_path):
    # Get the classifier dataset
    classifier_dataset = ADNIDataset(os.path.join(data_path, 'test'), classifier=True)
    return classifier_dataset


def get_classifier_test_dataset(data_path):
    # Get the classifier dataset
    classifier_dataset = ADNIDataset(os.path.join(data_path, 'test'), classifier=True)
    return classifier_dataset
