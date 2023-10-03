import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ADNIDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.root = root
        self.split = split  # 'train' or 'test'
        self.transform = transform

        self.image_paths = []
        self.labels = []

        # Define the classes (AD and NC)
        self.classes = ['AD', 'NC']

        # Load image paths and labels
        for class_name in self.classes:
            class_dir = os.path.join(root, split, class_name)
            class_label = self.classes.index(class_name)
            image_files = os.listdir(class_dir)
            for image_file in image_files:
                self.image_paths.append(os.path.join(class_dir, image_file))
                self.labels.append(class_label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx])

        return image, label