import torch
from torch.utils.data import DataLoader, Dataset
import os
from pathlib import Path
import sys
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import random
import math

ISIC_PATH = Path('/home', 'groups', 'comp3710', 'ISIC2018')
if sys.platform == 'win32':
    ISIC_PATH = Path('D:', 'ISIC2018')

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.25,), (0.25,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class ISIC(Dataset):
    def __init__(self, root_path, train=True, transform=None, image_files=None):
        self.path = root_path
        if train:
            self.path = Path(self.path, "ISIC2018_Task1-2_Training_Input_x2")
        else:
            self.path += "ISIC2018_Task1-2_Test_Input"
        self.transform = transform
        self.image_files = []
        if image_files is not None:
            self.image_files = image_files
        else:
            for filename in os.listdir(self.path):
                file_path = os.path.join(self.path, filename)
                self.image_files.append(file_path)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        label = os.path.split(img_path)[-1]
        return image, label

def split_train_val(dataset: ISIC, val_proportion: float):
    all_image_files = dataset.image_files
    split_point = math.floor(len(all_image_files)*val_proportion)
    random.shuffle(all_image_files)
    train_dataset = ISIC(root_path=ISIC_PATH, train=True, transform=None, image_files=all_image_files[:split_point])
    val_dataset = ISIC(root_path=ISIC_PATH, train=True, transform=None, image_files=all_image_files[split_point+1:])
    return train_dataset, val_dataset

def get_dataloader(batch_size, set="train", val_proportion: float = 0.2):
    train_dataset = ISIC(root_path=ISIC_PATH, transform=transform_train)
    train_dataset, val_dataset = split_train_val(dataset=train_dataset, val_proportion=val_proportion)
    loader = (DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(val_dataset, batch_size=batch_size, shuffle=True))
    if set == "test":
        dataset = ISIC(root_path=ISIC_PATH, transform=transform_test)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader