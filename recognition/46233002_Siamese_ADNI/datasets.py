from torch.utils.data import Dataset
import os
import cv2
import random
import numpy as np


class ADDataTrain(Dataset):

    def __init__(self, ad_dir, nc_dir, ads, ncs, transform=None):
        self.ad_dir = ad_dir
        self.nc_dir = nc_dir

        self.ads = sorted(ads)
        self.ncs = sorted(ncs)

        self.transform = transform

    def __len__(self):
        return len(self.ads) + len(self.ncs)

    def __getitem__(self, idx):
        # Determine if AD or NC 
        if idx < len(self.ads):
            path = self.ad_dir
            anchor_dir = self.ads
            pos_dir = self.ads
            neg_dir = self.ncs
            label = 1
        else:
            path = self.nc_dir
            anchor_dir = self.ncs
            pos_dir = self.ncs
            neg_dir = self.ads
            idx = idx - len(self.ads)
            label = 0
        
        # Read image
        # Positive and negative images are randomly sampled
        anchor = cv2.imread(os.path.join(path, anchor_dir[idx]))
        positive = cv2.imread(os.path.join(path, random.sample(pos_dir, 1)))
        negative = cv2.imread(os.path.join(path, random.sample(neg_dir, 1)))

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
        if idx < len(self.anchor_ads):
            path = self.ad_dir
            anchor_dir = self.anchor_ads
            pos_dir = self.ads
            neg_dir = self.ncs
            label = 1
        else:
            path = self.nc_dir
            anchor_dir = self.anchor_ncs
            pos_dir = self.ncs
            neg_dir = self.ads
            idx = idx - len(self.anchor_ads)
            label = 0
        
        # Read image 
        anchor = cv2.imread(os.path.join(path, anchor_dir[idx]))
        positive = cv2.imread(os.path.join(path, pos_dir[idx]))
        negative = cv2.imread(os.path.join(path, neg_dir[idx]))

        # Convert to np array and resize to single channel
        anchor = np.array(cv2.cvtColor(anchor, cv2.COLOR_BGR2GRAY))
        positive = np.array(cv2.cvtColor(positive, cv2.COLOR_BGR2GRAY))
        negative = np.array(cv2.cvtColor(negative, cv2.COLOR_BGR2GRAY))

        # Normalise
        anchor = anchor / 255.0
        positive = positive / 255.0
        negative = negative / 255.0

        return (anchor, positive, negative), label