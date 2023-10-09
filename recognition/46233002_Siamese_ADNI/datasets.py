from torch.utils.data import Dataset
import os
import cv2
import random
import numpy as np


class ADDataTrain(Dataset):

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
            anchor_path = os.path.join(self.ad_dir, self.ads[idx])
            pos_path = os.path.join(self.ad_dir, random.sample(self.ads, 1)[0])
            neg_path = os.path.join(self.nc_dir, random.sample(self.ncs, 1)[0])
            label = 1
        else:
            anchor_path = os.path.join(self.nc_dir, self.ncs[idx-len(self.ads)])
            pos_path = os.path.join(self.nc_dir, random.sample(self.ncs, 1)[0])
            neg_path = os.path.join(self.ad_dir, random.sample(self.ads, 1)[0])
            label = 0
        
        # Read image
        # Positive and negative images are randomly sampled
        anchor = cv2.imread(anchor_path)
        positive = cv2.imread(pos_path)
        negative = cv2.imread(neg_path)

        # Convert to np array and resize to single channel
        anchor = np.array(cv2.cvtColor(anchor, cv2.COLOR_BGR2GRAY))
        positive = np.array(cv2.cvtColor(positive, cv2.COLOR_BGR2GRAY))
        negative = np.array(cv2.cvtColor(negative, cv2.COLOR_BGR2GRAY))

        # Normalise
        anchor = anchor / 255.0
        positive = positive / 255.0
        negative = negative / 255.0

        return (anchor, positive, negative), label

class ADDataTest(Dataset):

    def __init__(self, ad_dir, nc_dir, ads, ncs, transform=None):
        self.ad_dir = ad_dir
        self.nc_dir = nc_dir

        self.anchor_ads = ads
        self.anchor_ncs = ncs

        self.transform = transform

        # Create copies for positive and negative image selection 
        self.ads = self.anchor_ads.copy()
        self.ncs = self.anchor_ncs.copy()

        # Sort anchors and randomize positive and negative images 
        self.anchor_ads = sorted(self.anchor_ads)
        self.anchor_ncs = sorted(self.anchor_ncs)
        random.shuffle(self.ads)
        random.shuffle(self.ncs)
    
    def __len__(self):
        return len(self.anchor_ads) + len(self.anchor_ncs)

    def __getitem__(self, idx):
        # Determine if AD or NC 
        if idx < len(self.ads):
            anchor_path = os.path.join(self.ad_dir, self.anchor_ads[idx])
            pos_path = os.path.join(self.ad_dir, self.ads[idx])
            neg_path = os.path.join(self.nc_dir, self.ncs[idx])
            label = 1
        else:
            anchor_path = os.path.join(self.nc_dir, self.anchor_ncs[idx-len(self.anchor_ads)])
            pos_path = os.path.join(self.nc_dir, self.ncs[idx-len(self.anchor_ads)])
            neg_path = os.path.join(self.ad_dir, self.ads[idx-len(self.anchor_ads)])
            label = 0
        
        # Read image 
        anchor = cv2.imread(anchor_path)
        positive = cv2.imread(pos_path)
        negative = cv2.imread(neg_path)

        # Convert to np array and resize to single channel
        anchor = np.array(cv2.cvtColor(anchor, cv2.COLOR_BGR2GRAY))
        positive = np.array(cv2.cvtColor(positive, cv2.COLOR_BGR2GRAY))
        negative = np.array(cv2.cvtColor(negative, cv2.COLOR_BGR2GRAY))

        # Normalise
        anchor = anchor / 255.0
        positive = positive / 255.0
        negative = negative / 255.0

        return (anchor, positive, negative), label