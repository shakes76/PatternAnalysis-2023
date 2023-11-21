import random

import torch
import os
import gc
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def load_images(folder, img_list):
    for filename in os.listdir(folder):
        with Image.open(os.path.join(folder, filename)) as img:
            img_list.append(img.copy())


class ADNISiameseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.normal_dir = os.path.join(data_dir, 'NC')
        self.ad_dir = os.path.join(data_dir, 'AD')

        self.triplets = []

        self.transform = transform

        self.preprocess()

    def preprocess(self):
        normal_images = []
        ad_images = []
        load_images(self.normal_dir, normal_images)
        load_images(self.ad_dir, ad_images)

        for i in range(len(ad_images)):
            pos_i = random.randint(0, len(ad_images) - 1)
            while pos_i == i:
                pos_i = random.randint(0, len(ad_images) - 1)
            neg_i = random.randint(0, len(normal_images) - 1)
            self.triplets.append((ad_images[i], ad_images[pos_i], normal_images[neg_i]))
        for i in range(len(normal_images)):
            pos_i = random.randint(0, len(normal_images) - 1)
            while pos_i == i:
                pos_i = random.randint(0, len(normal_images) - 1)
            neg_i = random.randint(0, len(ad_images) - 1)
            self.triplets.append((normal_images[i], normal_images[pos_i], ad_images[neg_i]))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative


class ADNIDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.normal_dir = os.path.join(data_dir, 'NC')
        self.ad_dir = os.path.join(data_dir, 'AD')
        self.normal_images = []
        self.ad_images = []
        load_images(self.normal_dir, self.normal_images)
        load_images(self.ad_dir, self.ad_images)

        self.transform = transform

    def __len__(self):
        return len(self.normal_images) + len(self.ad_images)

    def __getitem__(self, idx):
        if idx < len(self.normal_images):
            item = self.normal_images[idx]
            label = 0
        else:
            item = self.ad_images[idx - len(self.normal_images)]
            label = 1
        if self.transform:
            return self.transform(item), label
        else:
            return item, label
