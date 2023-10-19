"""
Created on Wednesday October 18 
ADNI Dataset and Data Loaders

This code defines a custom dataset class, ADNIDataset, for loading and processing
ADNI dataset images for use in Siamese Network training and testing. It also provides
functions to get train and test datasets from a specified data path.

@author: Aniket Gupta 
@ID: s4824063

"""
import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class SiameseADNIDataset(Dataset):
    def __init__(self, data_path, dataset_type):
        super(SiameseADNIDataset, self).__init()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
        ])

        ad_directory = os.path.join(data_path, dataset_type, 'AD')
        nc_directory = os.path.join(data_path, dataset_type, 'NC')

        self.ad_images = [self.load_and_transform_image(os.path.join(ad_directory, img)) for img in os.listdir(ad_directory)]
        self.nc_images = [self.load_and_transform_image(os.path.join(nc_directory, img)) for img in os.listdir(nc_directory)]

        self.ad_images = torch.stack(self.ad_images)
        self.nc_images = torch.stack(self.nc_images)

    def __len__(self):
        return min(len(self.ad_images), len(self.nc_images))

    def __getitem__(self, index):
        img1, img2, label = self.get_random_pair(index)
        return img1, img2, label

    def get_random_pair(self, index):
        img1 = self.ad_images[index % len(self.ad_images)]
        if index % 2 == 0:
            img2 = self.ad_images[(index + 1) % len(self.ad_images)]
            label = torch.tensor(1, dtype=torch.float)
        else:
            img2 = self.nc_images[index % len(self.nc_images)]
            label = torch.tensor(0, dtype=torch.float)

        return img1, img2, label

    def load_and_transform_image(self, img_path):
        img = Image.open(img_path)
        return self.transform(img)

def create_dataset(data_path, dataset_type):
    dataset = SiameseADNIDataset(data_path, dataset_type)
    return dataset
