import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from torchvision.transforms import ToTensor
# from torch.autograd import Function
# from itertools import repeat
# import numpy as np
# import os
# import pandas as pd
# from torchvision.io import read_image
import modules as m
import dataset as d
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Using cpu.")

# These are the hyper parameters for the training.
epochs = 30
learning_rate = 0.0001
batch = 32
gamma = 0.985

model = m.ModifiedUNet(3, 1).to(device)

img_dir = "ISIC2018_Task1-2_Training_Input_x2"
seg_dir = "ISIC2018_Task1_Training_GroundTruth_x2"
test_dir = "ISIC2018_Task1-2_Test_Input"
train_dataset = d.ISICDataset(img_dir, seg_dir, d.transform('train'), d.transform('seg'))
test_dataset = d.ISICDataset(test_dir, seg_dir)
train_loader = DataLoader(train_dataset, batch, shuffle=True)
test_loader = DataLoader(test_dataset, batch)

# We will use the ADAM optimizer
ADAMoptimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer=ADAMoptimizer, gamma=0.75)

# Now we begin timing
starttime = time.time()
for epoch in range(epochs):
    losslist = []
    # dicelist = []
    runningloss = 0.0


    model.train()

    for i, input in enumerate(train_loader):
        
        images, segments = input[0].to(device), input[1].to(device)

        ADAMoptimizer.zero_grad()

        modelled_images = model(images)[0]
        loss = m.dice_loss(modelled_images, segments)
        loss.backward()
        ADAMoptimizer.step()
        print(loss.item())
        runningloss += loss.item()
        print(runningloss)
    
    print(runningloss)