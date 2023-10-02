import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ISICDataset(Dataset):
    """ISIC Dataset class for loading and preprocessing data."""
    def __init__(self, dataset_type, transform=None):
        assert dataset_type in ['training', 'validation', 'test'], "Invalid dataset type. Must be one of ['training', 'validation', 'test']"
        self.root_dir = os.path.join('Data', dataset_type)  # Adjusting the root directory based on dataset type
        self.transform = transform
        self.image_list = sorted([f for f in os.listdir(self.root_dir) if f.endswith('.jpg')])
        self.mask_list = [f.replace('.jpg', '_superpixels.png') for f in self.image_list]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        mask_name = os.path.join(self.root_dir, self.mask_list[idx])
        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')
        sample = {'image': image, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
        return sample
