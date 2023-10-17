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
from torch.optim import lr_scheduler
import os
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pandas as pd 
import random

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")
else:
    print("Using CUDA.")

class Config():
    training_dir = "../AD_NC/train"
    testing_dir = "../AD_NC/test"
    train_batch_size = 8
    train_number_epochs = 20

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=.2),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=.2),
        )
        
        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(1966080, 500),
            #self.fc1 = nn.Linear(2*1000, 500)
            nn.Linear(500, 500),
            nn.Linear(500, 2)
        )
         
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
        for i, folder1 in enumerate(folders):
            for j, folder2 in enumerate(folders):
                c = 0
                print("Folder:", folder1, folder2)
                if i == j:
                    label = 0  # Images from the same folder
                else:
                    label = 1  # Images from different folders

                folder1_path = os.path.join(root_dir, folder1)
                folder2_path = os.path.join(root_dir, folder2)

                folder1_images = os.listdir(folder1_path)
                folder2_images = os.listdir(folder2_path)

                for img1 in folder1_images:
                    c += 1
                    if c % 1000 == 0:
                        print("Count:", c)
                    img2 = random.choice(folder2_images)
                    while img1 == img2:
                        print("FOUND SAME IMAGE - SHOULDN'T HAPPEN OFTEN")
                        img2 = random.choice(folder2_images)

                    img1_path = os.path.join(folder1_path, img1)
                    img2_path = os.path.join(folder2_path, img2)

                    self.image_paths.append((img1_path, img2_path))
                    self.labels.append(label)
                        
        print("< Finished creating image paths. #Images:", len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
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
    Based on formula provided during symposium.
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

# Decalre Loss Function
criterion = ContrastiveLoss()
optimizer = optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.99, eps=1e-8, weight_decay=0.0005, momentum=0.9)

def train():
    # counter = []
    # loss_history = [] 
    # iteration_number= 0
    
    for epoch in range(0,Config.train_number_epochs):
        print(enumerate(train_loader,0).__sizeof__(), Config.train_batch_size)
        print(enumerate(train_loader,0))
        for i, data in enumerate(train_loader,0):
            #Produce two sets of images with the label as 0 if they're from the same file or 1 if they're different
            print("i:", i, "/", int(43040 / Config.train_batch_size))
            img1, img2, labels = data
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            # if i < Config.train_batch_size - 1:
            # print(type(labels), labels.size(), labels[i])
            # print(type(images), images.size(), images[i])
            # imshow(images[i])
            # get tensor image
            # calculate mean and std
            # mean, std = images[i].mean([1,2]), images[i].std([1,2])
            
            # print mean and std
            # print("mean and std before normalize:")
            # print("Mean of the image:", mean)
            # print("Std of the image:", std)
            # print("Channels", torchvision.transforms.functional.get_image_num_channels(images[i]))
            # images = images.to(device)
            # labels = labels.to(device)
            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            # print("Out1len", output1.size(), "Out2len", output1.size(), "labelssize", labels.size())
            loss_contrastive = criterion(output1, output2, labels)
            loss_contrastive.backward()
            optimizer.step()
            if i % 1 == 0 :
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
                # iteration_number += 10
                # counter.append(iteration_number)
                # loss_history.append(loss_contrastive.item())
    return model

# Train the model
model = train()
torch.save(model.state_dict(), "model.pt")
print("Model Saved Successfully")