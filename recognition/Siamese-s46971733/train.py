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
batch_size = 32
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
print(f">>> Training \n")
for epoch in range(num_epochs):
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        # Extract data and transfer to GPU.
        anchor = data[0].to(device)
        positive = data[1].to(device)
        negative = data[2].to(device)

        # Zero the gradients -- Ensuring gradients not accumulated
        #                       across multiple training iterations.
        optimizer.zero_grad()

        # Forward Pass
        anchor_out = resnet(anchor)
        positive_out = resnet(positive)
        negative_out = resnet(negative)

        # Calculate Loss with Triplet Loss.
        loss = criterion(anchor_out, positive_out, negative_out)

        # Compute gradient with respect to model.
        loss.backward()

        # Optimizer step - Update model parameters.
        optimizer.step()

        running_loss += loss.item()

        # Print Loss Info while training.
        if (i + 1) % 10 == 0:
            print(f'[Epoch {epoch + 1}/{num_epochs}, {i + 1:5d}] - Loss: {running_loss / 10:.5f}')
            running_loss = 0.0