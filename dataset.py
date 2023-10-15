import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(os.path.join(data_dir, "images")) if f.endswith(".png")]
        self.mask_files = [f for f in os.listdir(os.path.join(data_dir, "masks")) if f.endswith(".png")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_dir, "images", self.image_files[index])
        mask_path = os.path.join(self.data_dir, "masks", self.mask_files[index])

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
