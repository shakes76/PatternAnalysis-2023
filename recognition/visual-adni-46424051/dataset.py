##################################   dataset.py   ##################################
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import random
import numpy as np

# Training set data loader
class DatasetTrain(Dataset):
    def __init__(self, path, transforms=None):
        super(DatasetTrain, self).__init__()
        self.transforms = transforms
        self.NC, self.AD = self.load_images(path)

    def load_images(self, path):
        NC = []
        AD = []
        
        # Load from filepath
        for filePath in os.listdir(os.path.join(path, "NC")):
            filePath = os.path.join(path, "NC", filePath)
            NC.append(Image.open(filePath).convert("L"))

        # Load from filepath
        for filePath in os.listdir(os.path.join(path, "AD")):
            filePath = os.path.join(path, "AD", filePath)
            AD.append(Image.open(filePath).convert("L"))

        return NC, AD

    # Size of epoch enumeration
    def __len__(self):
        return 10000
    
    def __getitem__(self, index):
        # Choose a random class and a random image from the folder
        image = None
        r = random.randint(0, 1)
        if r == 0:
            image = random.choice(self.NC)
        if r == 1:
            image = random.choice(self.AD)
        
        # Transform size and to tensor and return with label
        if self.transforms:
            image = self.transforms(image)
        return image, torch.tensor(r, dtype=torch.float32)
    
# Testing set data loader
class DatasetTest(Dataset):
    def __init__(self, path, transforms=None, size=1000):
        super(DatasetTest, self).__init__()
        self.transforms = transforms
        self.size = size
        self.NC, self.AD = self.load_images(path)
        self.classy = 0
        self.NCIndex = 0
        self.ADIndex = 0

    def load_images(self, path):
        NC = []
        AD = []
        
        # Load from filepath
        for filePath in os.listdir(os.path.join(path, "NC")):
            filePath = os.path.join(path, "NC", filePath)
            NC.append(Image.open(filePath).convert("L"))

        # Load from filepath
        for filePath in os.listdir(os.path.join(path, "AD")):
            filePath = os.path.join(path, "AD", filePath)
            AD.append(Image.open(filePath).convert("L"))

        return NC, AD

    # Size of epoch enumeration
    def __len__(self):
        return len(self.NC) + len(self.AD)
    
    def __getitem__(self, index):
        # Cycle through each image in first NC then AD
        image = None
        r = self.classy
        if r == 0:
            if self.NCIndex == len(self.NC) - 1:
                self.classy = 1
            image = self.NC[self.NCIndex]
            self.NCIndex+=1
        if r == 1:
            image = self.AD[self.ADIndex]
            self.ADIndex+=1
        
        # Transform size and to tensor and return with label
        if self.transforms:
            image = self.transforms(image)
        return image, torch.tensor(r, dtype=torch.float32)
