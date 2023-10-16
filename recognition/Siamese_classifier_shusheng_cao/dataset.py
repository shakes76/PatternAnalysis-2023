import random

import torch
import os
import gc
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ADNITrainDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.normal_dir = os.path.join(data_dir, '/NC/')
        self.ad_dir = os.path.join(data_dir, '/AD/')
        self.normal_images_file = [os.path.join(self.data_dir, path) for path in os.listdir(self.normal_dir)]
        self.ad_images_file = [os.path.join(self.data_dir, path) for path in os.listdir(self.ad_dir)]
        self.normal_image = torch.tensor([Image.open(img_path) for img_path in self.normal_images_file])
        self.ad_image = torch.tensor([Image.open(img_path) for img_path in self.ad_images_file])
        self.triplets = []

        self.transform = self.compute_mean_std()

        self.preprocess()

    def preprocess(self):
        ad_triplets = [(anchor, positive, negative)
                       for anchor in self.ad_image
                       for positive in random.choices([x for x in self.ad_image if x != anchor], k=1)
                       for negative in random.choices(self.normal_image, k=1)]

        nc_triplets = [(anchor, positive, negative)
                       for anchor in self.ad_image
                       for positive in random.choices([x for x in self.normal_image if x != anchor], k=1)
                       for negative in random.choices(self.ad_image, k=1)]
        self.triplets += ad_triplets + nc_triplets

    def compute_mean_std(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        total_images = torch.Tensor(self.ad_image + self.normal_image)
        total_images.to(device)
        mean = total_images.mean(2).sum(0)
        std = total_images.std(2).sum(0)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        del total_images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return transform

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative
