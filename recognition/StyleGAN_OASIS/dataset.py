# -*- coding: utf-8 -*-
"""
File: dataset.py

Purpose: Tools required to load and preprocess the data

@author: Peter Beardsley
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

"""
Create a data loader:
    image_folder    references the folder containing the classes of images
    image_size      resize the images to this size
    batch_size      the number of images the data loader should batch

Returns:
    data loader
    
Notes: When training a progressive GAN, a dataloader is required for each
       progressive image size. As image size increases, the batch size must
       decrease to accommodate the increased memory requirements.
"""
def createDataLoader(image_folder, image_size, batch_size):
  
    transform = transforms.Compose([transforms.Resize((image_size, image_size))
                                    , transforms.ToTensor()
                                    , transforms.Normalize((0.5,), (0.5,))])

    dataset = datasets.ImageFolder(root=image_folder, transform=transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader



"""
Get the size of a dataset:
    image_folder    references the folder containing the classes of images
    
Returns:
    Number of images in the dataset
"""
def getDataSize(image_folder):
    return len(datasets.ImageFolder(root=image_folder))