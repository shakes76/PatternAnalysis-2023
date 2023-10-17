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


class ADNITrainSiameseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.normal_dir = os.path.join(data_dir, 'NC')
        self.ad_dir = os.path.join(data_dir, 'AD')
        self.normal_images = []
        self.ad_images = []
        load_images(self.normal_dir, self.normal_images)
        load_images(self.ad_dir, self.ad_images)

        self.triplets = []

        self.transform = transform

        self.preprocess()

    def preprocess(self):
        for i in range(len(self.ad_images)):
            pos_i = random.randint(0, len(self.ad_images) - 1)
            while pos_i == i:
                pos_i = random.randint(0, len(self.ad_images) - 1)
            neg_i = random.randint(0, len(self.normal_images) - 1)
            self.triplets.append((self.ad_images[i], self.ad_images[pos_i], self.normal_images[neg_i]))
        for i in range(len(self.normal_images)):
            pos_i = random.randint(0, len(self.normal_images) - 1)
            while pos_i == i:
                pos_i = random.randint(0, len(self.normal_images) - 1)
            neg_i = random.randint(0, len(self.ad_images) - 1)
            self.triplets.append((self.normal_images[i], self.normal_images[pos_i], self.ad_images[neg_i]))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative


class ADNITrainClassifierDataset(Dataset):
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
        else:
            item = self.ad_images[idx - len(self.normal_images)]
        if self.transform:
            return self.transform(item)
        else:
            return item
