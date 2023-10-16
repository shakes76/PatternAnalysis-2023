import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import PIL.ImageOps
import torch

class CustomDataset(Dataset):

    def __init__(self, image_dataset, transform=None, should_invert=True):
        self.image_dataset = image_dataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img1_tuple = random.choice(self.image_dataset.imgs)
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                img2_tuple = random.choice(self.image_dataset.imgs)
                if img1_tuple[1] == img2_tuple[1]:
                    break
        else:
            while True:
                img2_tuple = random.choice(self.image_dataset.imgs)
                if img1_tuple[1] != img2_tuple[1]:
                    break

        img1 = Image.open(img1_tuple[0])
        img2 = Image.open(img2_tuple[0])
        img1 = img1.convert("L")
        img2 = img2.convert("L")

        if self.should_invert:
            img1 = PIL.ImageOps.invert(img1)
            img2 = PIL.ImageOps.invert(img2)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.from_numpy(np.array([int(img1_tuple[1] != img2_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.image_dataset.imgs)
