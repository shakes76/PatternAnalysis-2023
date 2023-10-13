"""
dataset.py

Student Name: Zijun Zhu
Student ID: s4627546
Bref intro:
containing the data loader for loading and preprocessing the data
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class SuperResolutionDataset(Dataset):
    """
    Used for 3710- ADNI_AD_NC_2D dataset
    This class will load dataset:
    (lr_image, hr_image)
    lr_image: low resolution image,64x60   , downsample by factor of 4
    hr_image: high resolution image,256x240, original image in dataset
    """

    def __init__(self, root_dir, transform=None, mode='train'):
        """
        Initialize the dataset.
        :param root_dir (str): The dir of dataset folder
        :param transform:      transform to apply on a sample.
        :param mode    (str): 'train' or 'test' data to load.
        """
        self.root_dir = root_dir
        self.transform =transform if transform is not None else transforms.Compose([transforms.ToTensor()])
        # default: convert images to tensors
        self.mode = mode
        self.AD_paths = sorted(os.listdir(os.path.join(root_dir, mode, 'AD')))
        self.NC_paths = sorted(os.listdir(os.path.join(root_dir, mode, 'NC')))

    def __len__(self):
        return len(self.AD_paths) + len(self.NC_paths)

    def __getitem__(self, idx):
        """
        :param idx:
                train: 0-10399 for AD 10400-11119 for NC
                test: 0-4459   for AD 4460-4539   for NC
        :return: (lr_image, hr_image)
                lr_image: low resolution image 64x60, Downsample by factor of 4
        """
        if idx < len(self.AD_paths):
            img_path = os.path.join(self.root_dir, self.mode, 'AD', self.AD_paths[idx])
        else:
            img_path = os.path.join(self.root_dir, self.mode, 'NC', self.NC_paths[idx - len(self.AD_paths)])

        hr_image = Image.open(img_path).convert('L')
        # Convert image to grayscale, although all data in ADNI_dataset are 2D
        lr_transform = transforms.Compose([transforms.Resize((64, 60), interpolation=Image.BICUBIC)])
        # Downsample by factor of 4
        lr_image = lr_transform(hr_image)

        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        return lr_image, hr_image


if __name__ == '__main__':
    train_dataset = SuperResolutionDataset(root_dir='AD_NC', transform=None, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = SuperResolutionDataset(root_dir='AD_NC', transform=None, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
