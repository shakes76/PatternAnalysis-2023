
"""Classes for creating dataloaders of grayscale images from folders"""


import os
import torch
import torchvision.transforms as transforms
import numpy as np  
from tqdm.auto import tqdm
from PIL import Image
import torch.utils.data


"""Prepares train, validation and test datasets"""
# Note, making datasets class variables can prevent needing to reload data every instance
# (or add conditional to see if data is previously loaded in init())
class DataPreparer():

    def __init__(self, path, train_folder, validation_folder, test_folder, batch_size):
        # Transform (to be applied to all datasets)
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=0.13242, std=0.18826)
        ])
        self.batch_size = batch_size
        # Initialise the datasets
        train_data = self.load_data_from_folder(path, train_folder)
        validate_data = self.load_data_from_folder(path, validation_folder)
        test_data = self.load_data_from_folder(path, test_folder)

        # Transform the data, and stack it into a tensor (B, C, H, W)
        self.train_dataset = torch.stack([transform(item) for item in train_data]).permute(0, 2, 3, 1)
        self.validate_dataset = torch.stack([transform(item) for item in validate_data]).permute(0, 2, 3, 1)
        self.test_dataset = torch.stack([transform(item) for item in test_data]).permute(0, 2, 3, 1)

        # Create dataloaders
        self.train_dataloader = self.prepare_dataset(self.train_dataset)
        self.validate_dataloader = self.prepare_dataset(self.validate_dataset)
        self.test_dataloader = self.prepare_dataset(self.test_dataset)

    """Takes in a dataset, returns a dataloader"""
    def prepare_dataset(self, dataset):
        # Create and return dataloader
        dataloader = torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=True)
        return dataloader

    """Given a path and a folder name, returns a numpy array of the images (B, C=1, H, W)"""
    def load_data_from_folder(self, path, name):
        # Initialise data as an empty list
        data = []
        i = 0
        # Make a list of image names
        list_files = [f for f in os.listdir(path+name) if f.lower().endswith('.png')]

        for filename in tqdm(list_files):
            # Load image as numpy array
            image_path = os.path.join(path+name, filename)
            # Assign and convert to grayscale
            image = Image.open(image_path).convert('L')
            image = np.array(image)
            # Add channel, so img shape is 'C=1, H, W'
            image = np.expand_dims(image, axis=0)

            data.append(image)

            if i == 50:
                return data
            i += 1

        return np.array(data)