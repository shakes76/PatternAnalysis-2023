"""
Author: Zach Harbutt S4585714
contains the data loader for loading and preprocessing ADNI dataset

ADNI: https://adni.loni.usc.edu/
"""

import torch
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# hyper-parameters
batch_size = 32
train_prop = 0.9

class ADNIDataset(Dataset):
    """
    Loads Dataset from 3710 - ADNI_AD_NC_2D data
    """
    def __init__(self, root, transform, mode='train', image_size=(240,256), downscale_factor=4):
        super(ADNIDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.mode = mode
        self.image_size = image_size
        self.downscale_factor = downscale_factor
        self.AD = sorted(os.listdir(os.path.join(self.root, self.mode, 'AD')))
        self.NC = sorted(os.listdir(os.path.join(self.root, self.mode, 'NC'))) 
        
    def __getitem__(self, index):
        if index < len(self.AD):
            path = os.path.join(self.root, self.mode, 'AD', self.AD[index])
        else:
            path = path = os.path.join(self.root, self.mode, 'NC', self.NC[index-len(self.AD)])
        
        orig = Image.open(path)
        downscaler = transforms.Compose([
            transforms.Resize((int(self.image_size[0]/self.downscale_factor), 
                               int(self.image_size[1]/self.downscale_factor)), 
                              interpolation=Image.BICUBIC) 
            ])
        downscaled = downscaler(orig)
        
        downscaled = self.transform(downscaled)
        orig = self.transform(orig)
        
        return downscaled, orig
            
        
        
    def __len__(self):
        return len(self.AD) + len(self.NC)

def ADNIDataLoader(root, mode='train'):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        ])
    dataset = ADNIDataset(root, transform, mode=mode)
    
    if (mode == 'train'):
        trainset, validset = random_split(dataset, [int(len(dataset) * train_prop), len(dataset) - int(len(dataset) * train_prop)])
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=False)
        return train_loader, valid_loader
    else:
        return DataLoader(dataset, batch_size=1, shuffle=False)