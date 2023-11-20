"""
    File name: dataset.py
    Author: Yicheng Jia
    Date created: 27/09/2023
    Date last modified: 21/11/2023
    Python Version: 3.11.04
"""


import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class SiameseNetworkDataset(Dataset):
    def __init__(self, root_dir, classifier=False):
        super(SiameseNetworkDataset, self).__init__()

        self.classifier = classifier

        """
        I used to use the following ways to increase the generalization ability of module.
        But the dataset is too simple that it doesn't require so many pre operations.
        So I just give them up.
        
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(degrees=(0, 60)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
            transforms.RandomPerspective(),
        """
        self.transform = transforms.Compose([
            transforms.ToTensor(),

            # No need to do this normalization
            # transforms.Normalize(mean=[0.485], std=[0.229])
        ])

        # Define categories
        self.categories = ['AD', 'NC']

        # Load images for each category
        self.images = {category: [self.transform(Image.open(os.path.join(root_dir, category, img_filename)))
                                  for img_filename in os.listdir(os.path.join(root_dir, category))
                                  if os.path.exists(os.path.join(root_dir, category, img_filename))]
                       for category in self.categories}

        # Stack images into tensors for each category
        for category in self.categories:
            self.images[category] = torch.stack(self.images[category])

    # Get the length of dataset
    def __len__(self):
        if not self.classifier:
            return min(len(self.images['AD']), len(self.images['NC']))
        else:
            return 2 * min(len(self.images['AD']), len(self.images['NC']))

    def __getitem__(self, index):
        if not self.classifier:
            if index % 2 == 0:
                # Positive example (both images are AD)
                img1 = self.images['AD'][index % len(self.images['AD'])]
                img2 = self.images['AD'][index % len(self.images['AD'])]
                label = torch.tensor(1, dtype=torch.float)
            else:
                # Negative example (one image is AD, the other is NC)
                img1 = self.images['AD'][index % len(self.images['AD'])]
                img2 = self.images['NC'][index % len(self.images['NC'])]
                label = torch.tensor(0, dtype=torch.float)

            return img1, img2, label
        else:
            if index % 2 == 0:
                img = self.images['AD'][index % len(self.images['AD'])]
                label = torch.tensor(1, dtype=torch.float)
            else:
                img = self.images['NC'][index % len(self.images['NC'])]
                label = torch.tensor(0, dtype=torch.float)
            return img, label


def get_datasets(root_dir):
    # Load the datasets
    train_dataset = SiameseNetworkDataset(os.path.join(root_dir, 'train'))
    test_dataset = SiameseNetworkDataset(os.path.join(root_dir, 'test'))
    val_dataset = SiameseNetworkDataset(os.path.join(root_dir, 'val'))
    return train_dataset, test_dataset, val_dataset


def get_classifier_train_dataset(data_path):
    # Get the classifier dataset
    classifier_train_dataset = SiameseNetworkDataset(os.path.join(data_path, 'train'), classifier=True)
    return classifier_train_dataset


def get_classifier_test_dataset(data_path):
    # Get the classifier dataset
    classifier_test_dataset = SiameseNetworkDataset(os.path.join(data_path, 'test'), classifier=True)
    return classifier_test_dataset
