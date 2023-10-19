"""
Created on Wednesday October 18 
ADNI Dataset and Data Loaders

This code defines a custom dataset class, ADNIDataset, for loading and processing
ADNI dataset images for use in Siamese Network training and testing. It also provides
functions to get train and test datasets from a specified data path.

@author: Aniket Gupta 
@ID: s4824063

"""

from torchvision import transforms
import os
from PIL import Image
import torch
from torch.utils.data import Dataset


# Define a custom dataset class for SiameseADNIDataset
class SiameseADNIDataset(Dataset):
    def __init__(self, data_path, dataset_type):
        super(SiameseADNIDataset, self).__init__()
    # Define a transformation to convert images to tensors
        self.transform = transforms.ToTensor()
    # Create paths to AD and NC image directories
        ad_directory = os.path.join(data_path, dataset_type, 'AD')
        nc_directory = os.path.join(data_path, dataset_type, 'NC')
    # Load and transform AD images
        self.ad_images = [self.load_and_transform_image(os.path.join(ad_directory, img)) for img in os.listdir(ad_directory)]
    # Load and transform NC images
        self.nc_images = [self.load_and_transform_image(os.path.join(nc_directory, img)) for img in os.listdir(nc_directory)]
    # Stack AD and NC images into tensors
        self.ad_images = torch.stack(self.ad_images)
        self.nc_images = torch.stack(self.nc_images)
    # Function to load and transform an image
    def load_and_transform_image(self, img_path):
        img = Image.open(img_path)
        return self.transform(img)
    # Function to get the length of the dataset
    def __len__(self):
        return min(len(self.ad_images), len(self.nc_images))
    # Function to get items (image pairs and labels) from the dataset
    def __getitem__(self, index):
        img1 = self.ad_images[index % len(self.ad_images)]
        if index % 2 == 0:
            img2 = self.ad_images[(index + 1) % len(self.ad_images)]
            label = torch.tensor(1, dtype=torch.float)
        else:
            img2 = self.nc_images[index % len(self.nc_images)]
            label = torch.tensor(0, dtype=torch.float)

        return img1, img2, label

# Function to create a dataset
def create_dataset(data_path, dataset_type):
    dataset = SiameseADNIDataset(data_path, dataset_type)
    return dataset
# Function to get the training dataset
def get_training(data_path):
    return create_dataset(data_path, 'train')
# Function to get the testing dataset
def get_testing(data_path):
    return create_dataset(data_path, 'test')