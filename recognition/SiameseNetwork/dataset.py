# containing the data loader for loading and preprocessing your data

import os
import torch
import random
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, ad_dir, nc_dir, transform=None):
        # get the file path
        self.ad_folder = ad_dir
        self.nc_folder = nc_dir

        # get the samples' name
        self.ad_names = os.listdir(ad_dir)
        self.nc_names = os.listdir(nc_dir)

        # define the transform
        self.transform = transform

    def __len__(self):
        return 2 * min(len(self.AD_names), len(self.NC_names))

    def __getitem__(self, index):

        # path to the image, randomly select ad image and nc image
        # chooce anchor image evently from ad image set and nc image set
        if index % 2 == 0:
            anchor_path = os.path.join(self.ad_folder, self.ad_names[index//2])
        else:
            anchor_path = os.path.join(self.nc_folder, self.nc_names[index//2])
        ad_path = os.path.join(self.ad_folder, random.choice(self.ad_names))
        nc_path = os.path.join(self.nc_folder, random.choice(self.nc_names))

        # open images
        anchor = Image.open(anchor_path)
        ad = Image.open(ad_path)
        nc = Image.open(nc_path)

        # apply transformation
        if self.transform:
            anchor = self.transform(anchor)
            ad = self.transform(ad)
            nc = self.transform(nc)
        
        return anchor, ad, nc