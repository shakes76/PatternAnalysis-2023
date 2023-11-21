"""
Swin Transformer Model Architecture Based Topic Recognition for Alzheimer's Disease Classification
Name: Tarushi Gera
Student ID: 48242204
This code is to make training and testing easier by loading and preprocessing images from a directory.
"""
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class AlzheimerDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.class_names = os.listdir(data_dir)
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for filename in os.listdir(class_dir):
                image_path = os.path.join(class_dir, filename)
                data.append((image_path, class_idx))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
