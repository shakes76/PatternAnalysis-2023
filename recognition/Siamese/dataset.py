#This is for setting up the custom dataloader for the siamese model

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision.datasets as datasets
import torch.utils.data as util_data
from torch.utils.data import DataLoader, Dataset
import torch
import random
from PIL import Image

train_path = "C:\\Users\\Asus\\Desktop\\AD_NC\\train"
test_path = "C:\\Users\\Asus\\Desktop\\AD_NC\\test"

transform = transforms.Compose([transforms.ToTensor()])

class Siamese_dataset(Dataset):
    def __init__(self, imageFolder, transform):
        self.imageFolder = imageFolder
        self.transform = transform
    
    def __len__(self):
        return len(self.imageFolder.imgs)

    def __getitem__(self, index):
        #we need two images in order to compare similarity
        img0 = random.choice(self.imageFolder.imgs)

        #First we decide if were selecting two images with different classes or the same
        same_class = random.randint(0,1)

        if same_class: #if 1 then select two images with the same class
            while True:
                img1 = random.choice(self.imageFolder.imgs)
                if img0[1] == img1[1]:
                    break
        
        else:
            while True:
                img1 = random.choice(self.imageFolder.imgs)
                if img0[1] != img1[1]:
                    break
        
        img0_Image = Image.open(img0)
        img1_Image = Image.open(img1)

        img0_Image = img0_Image.convert("L")
        img1_Image = img1_Image.convert("L")

        img0_Image = self.transform(img0_Image)
        img1_Image = self.transform