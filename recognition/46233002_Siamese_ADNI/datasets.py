from torch.utils.data import Dataset
import os
import cv2
import numpy as np



class adniDataset(Dataset):

    def __init__(self, ad_dir, nc_dir, ads, ncs, transform=None):
        self.ad_dir = ad_dir # /home/groups/comp3710/ADNI/AD_NC/train/AD
        self.nc_dir = nc_dir # /home/groups/comp3710/ADNI/AD_NC/train/NC

        self.ads = sorted(ads)
        self.ncs = sorted(ncs)

        self.transform = transform

    def __len__(self):
        return len(self.ads) + len(self.ncs)

    def __getitem__(self, idx):
        # Determine if AD or NC 
        if idx < len(self.ads):
            path = os.path.join(self.ad_dir, self.ads[idx])
            label = 1
        else:
            path = os.path.join(self.nc_dir, self.ncs[idx-len(self.ads)])
            label = 0
        
        # Read image
        # Positive and negative images are randomly sampled
        image = cv2.imread(path)

        # Convert to np array and resize to single channel
        image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        # Normalise
        image = image / 255.0

        # Transformation
        if self.transform != None:
            image = self.transform(image)

        return image, label