"""
Created on Monday Sep 18 12:20:00 2023

This script is for loading in the downloaded OASIS Brain data set(9K images), creating a data loader 
and performing any required preprocessing it before training. 
The data is a preprocessed version of the original OASIS Brain dataset provided by COMP3710 course staff.

@author: Gabriel Russell
@ID: s4640776

"""
import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset
from modules import Parameters

"""
This class takes the downloaded data from a specific path, performs the required 
transformation to all images and returns it ready to be loaded into a dataloader
"""
class OASISDataset(Dataset):
    #Define the transform as a class attribute

    #Define the data transformations from imported datasets
    #Resize all images to 256x256 pixels
    #Convert to tensors and normalise tensor image with mean and standard deviation
    transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])])

    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.data = os.listdir(data_directory)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_directory, self.data[idx])
        img = Image.open(img_path)
        img = self.transform(img)
        return img

"""
This class calls the OASISDataset Class in it's initialisations to set up 
the train, validation and test datasets.
Included are some getter functions for returning the specified dataloader.
"""
class OASISDataloader():
    def __init__(self):
        p = Parameters()
        current_dir = os.getcwd()
        OASIS_train_path = current_dir + '\keras_png_slices_train'
        OASIS_validate_path = current_dir + '\keras_png_slices_validate'
        OASIS_test_path = current_dir + '\keras_png_slices_test'
        self.train = OASISDataset(OASIS_train_path)
        self.validate = OASISDataset(OASIS_validate_path)
        self.test = OASISDataset(OASIS_test_path)
        self.batch_size = p.batch_size
        self.train_dataloader =  DataLoader(self.train, batch_size = self.batch_size, shuffle = True, drop_last= True)

    def get_train(self):
        self.train_dataloader =  DataLoader(self.train, batch_size = self.batch_size, shuffle = True, drop_last= True)
        return self.train_dataloader
    
    def get_validate(self):
        validate_dataloader =  DataLoader(self.validate, batch_size = self.batch_size, shuffle = False)
        return validate_dataloader
    
    def get_test(self):
        test_dataloader =  DataLoader(self.test, batch_size = self.batch_size, shuffle = False)
        return test_dataloader
    