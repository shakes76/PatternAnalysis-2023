# Contains the source code of the components of your model. Each component must be
# implementated as a class or a function
# Import all the necessary libraries
import torchvision
import torch.utils.data as utils
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import time
import copy
from torch.optim import lr_scheduler
import os
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pandas as pd 
import random

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

class Config():
    training_dir = "../AD_NC/train"
    testing_dir = "../AD_NC/test"
    train_batch_size = 8
    train_number_epochs = 1

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            
            nn.Conv2d(3, 96, kernel_size=11,stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(p=0.3),

            nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(p=0.3),

        )
        
        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(200448, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128,2))
        
  
    def forward_once(self, x):
        # Forward pass 
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2
    
class CustomSiameseNetworkDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Create a list of image paths
        self.image_paths = []
        self.labels = []
        folders = os.listdir(root_dir)
        print("> Creating image paths")
        c = 0
        for i, folder1 in enumerate(folders):
            for j, folder2 in enumerate(folders):
                print("Folder:", folder1, folder2)
                if i == j:
                    label = 0  # Images from the same folder
                else:
                    label = 1  # Images from different folders

                folder1_path = os.path.join(root_dir, folder1)
                folder2_path = os.path.join(root_dir, folder2)

                for img1 in os.listdir(folder1_path):
                    c += 1
                    if c % 1000 == 0:
                        print("Count:", c)
                    img2 = random.choice(os.listdir(folder2_path))
                    while img1 == img2:
                        img2 = random.choice(os.listdir(folder2_path))

                img1_path = os.path.join(folder1_path, img1)
                img2_path = os.path.join(folder2_path, img2)

                self.image_paths.append((img1_path, img2_path))
                self.labels.append(label)
                        
        print("< Finished creating image paths. #Images:", len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        print("Getting item")
        img1_path, img2_path = self.image_paths[index]
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = torch.tensor(self.labels[index], dtype=torch.float32)
        
        return img1, img2, label
    
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
    
transform_train = transforms.Compose([
    transforms.ToTensor(),
])
    
# trainset = datasets.ImageFolder('C:\\Users\\david\\OneDrive\\Documents\\0NIVERSITY\\2023\\SEM2\\COMP3710\\Project\\PatternAnalysis-2023\\recognition\\Siamese_45330212\\AD_NC\\train', transform=transform_train)
trainset = CustomSiameseNetworkDataset(root_dir='C:\\Users\\david\\OneDrive\\Documents\\0NIVERSITY\\2023\\SEM2\\COMP3710\\Project\\PatternAnalysis-2023\\recognition\\Siamese_45330212\\AD_NC\\train', transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=Config.train_batch_size, shuffle=True)

testset = datasets.ImageFolder('C:\\Users\\david\\OneDrive\\Documents\\0NIVERSITY\\2023\\SEM2\\COMP3710\\Project\\PatternAnalysis-2023\\recognition\\Siamese_45330212\\AD_NC\\test', transform=transform_train)
test_loader = torch.utils.data.DataLoader(testset, batch_size=Config.train_batch_size, shuffle=True)

model = SiameseNetwork()
model = model.to(device)