import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from modules import ImprovedUNet
from dataset import CustomDataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Hyper-parameters
num_epochs = 1
learning_rate = 5e-3
channels = 10

#--------------
#Data
transform_train = transforms.Compose([transforms.ToTensor()])
transform_test = transforms.Compose([transforms.ToTensor()])


trainset = CustomDataset('C:\\Users\\JRSan\\Downloads\\ISIC2018\\ISIC2018_Task1-2_Training_Input_x2',
                         "C:\\Users\\JRSan\\Downloads\\ISIC2018\\ISIC2018_Task1_Training_GroundTruth_x2" ,transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
total_step = len(train_loader)

#Training
model = ImprovedUNet(in_channels=3, n_classes=2)
model = model.to(device)

#model info
print("Model No. of Parameters:", sum([param.nelement() for param in model.parameters()]))
print(model)

#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

#Piecewise Linear Schedule
total_step = len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_step*num_epochs)

#--------------
# Train the model
model.train()
print("> Training")
start = time.time() #time generation
for epoch in range(num_epochs):
    for i, (images, masks) in enumerate(train_loader): #load a batch
        images = images.to(device)
        masks = masks.to(device)

        print(images.dim())

        images = images.permute(1, 0, 2, 3)
        masks = masks.permute(1, 0, 2, 3)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 5 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}"
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
        scheduler.step()
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total") 
