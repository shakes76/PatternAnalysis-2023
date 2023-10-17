"""
Author: Zach Harbutt S4585714
contains the data loader for loading and preprocessing ADNI dataset

ADNI: https://adni.loni.usc.edu/
"""

import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ADNIDataset(Dataset):
    """
    Loads Dataset from 3710 - ADNI_AD_NC_2D data
    """
    def __init__(self, root, mode='train', image_size=(256,240), downscale_factor=4):
        super(ADNIDataset, self).__init__()
        self.root = root
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
                               int(self.image_size[1]/self.downscale_factor)))
            ])
        downscaled = downscaler(orig)
        
        to_tensor = transforms.Compose ([
            transforms.ToTensor()
            ])
        downscaled = to_tensor(downscaled)
        orig = to_tensor(orig)
        
        return downscaled, orig
            
        
        
    def __len__(self):
        return len(self.AD) + len(self.NC)
        
