# Imports
import time
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import os
from dataset import get_dataset
from modules import Resnet, Resnet34

num_epochs = 10
batch_size = 64
learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("No CUDA Found. Using CPU")

print("\n")

# Datasets and Dataloaders
trainset = get_dataset(train=1)
testset = get_dataset(train=0)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

# Model.
resnet = Resnet().to(device)

# Optimizer
criterion = nn.TripletMarginLoss()
optimizer = optim.Adam(resnet.parameters(), lr=learning_rate)

# Future spot for Scheduler?


#

for epoch in range(num_epochs):
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        # Extract data and transfer to GPU.
        anchor = data[0].to(device)

        # Zero the gradients -- Ensuring gradients not accumulated
        #                       across multiple training iterations.
        optimizer.zero_grad()

        # Forward Pass
        anchor_out = resnet(anchor)

        # Calculate Loss with Triplet Loss.
        #loss = criterion(anchor_out, positive_out, negative_out)
        # Compute gradient with respect to model.
        #loss.backward()

        # Optimizer step - Update model parameters.
        optimizer.step()

        running_loss += loss.item()
        print(f"Epoch: {epoch+1} / {num_epochs} - Loss: {running_loss}")