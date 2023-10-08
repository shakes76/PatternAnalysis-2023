##################################   dataset.py   ##################################
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import random
import numpy as np

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

    def __len__(self):
        return 11120
    
    def __getitem__(self, index):
        image1 = None
        image2 = None
        r = random.randint(0, 2)
        if r == 0:
            image1 = random.choice(self.NC)
            image2 = random.choice(self.NC)
        if r == 1:
            image1 = random.choice(self.NC)
            image2 = random.choice(self.AD)
        if r == 2:
            image1 = random.choice(self.AD)
            image2 = random.choice(self.AD)
        
        if self.transforms:
            image1 = self.transforms(image1)
            image2 = self.transforms(image2)
        return image1, image2, torch.tensor(r % 2, dtype=torch.float32)
    
class DatasetTest(Dataset):
    def __init__(self, path, transforms=None, size=1000):
        super(DatasetTest, self).__init__()
        self.transforms = transforms
        self.size = size
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

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        image1 = None
        image2 = None
        r = random.randint(0, 1)
        if r == 0:
            image1 = random.choice(self.NC)
            image2 = random.choice(self.NC)
        if r == 1:
            image1 = random.choice(self.NC)
            image2 = random.choice(self.AD)
        if r == 2:
            image1 = random.choice(self.AD)
            image2 = random.choice(self.AD)
        
        if self.transforms:
            image1 = self.transforms(image1)
            image2 = self.transforms(image2)
        return image1, image2, torch.tensor(r % 2, dtype=torch.float32)
    
if __name__=="__main__":
    dataset = DatasetTrain(os.path.expanduser("~/../../groups/comp3710/ADNI/AD_NC/train"), None)
    dataset = DatasetTest(os.path.expanduser("~/../../groups/comp3710/ADNI/AD_NC/test"), None)
