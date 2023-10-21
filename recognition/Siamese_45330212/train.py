""" Contains the source code for training, validating, testing and saving your model. The model
should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
sure to plot the losses and metrics during training """
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
import time
from modules import *
from dataset import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")
else:
    print("Using CUDA.")

# model = SiameseNetwork()
model = ResNet18()
model = model.to(device)

print("Model No. of Parameters:", sum([param.nelement() for param in model.parameters()]))

# Decalre Loss Function
criterion = ContrastiveLoss()
criterion_triplet = TripletLoss()
learning_rate = 1e-4
# optimizer = optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.99, eps=1e-8, weight_decay=0.0005, momentum=0.9)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_loader), epochs=Config.train_number_epochs)

# --------------
# Train the model
model.train()
print("> Training")
start = time.time() #time generation
counter = []
loss_history = []
iteration_number= 0
# print(sum(p.numel() for p in model.parameters() if p.requires_grad))
def siameseTwinTraining():
    for epoch in range(0,Config.train_number_epochs):
        for i, data in enumerate(train_loader,0):
            #Produce two sets of images with the label as 0 if they're from the same file or 1 if they're different
            img1, img2, labels = data
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            # Forward pass
            output1, output2 = model(img1, img2)
            # print("Out1len", output1.size(), "Out2len", output1.size(), "labelssize", labels.size())
            loss_contrastive = criterion(output1, output2, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss_contrastive.backward()
            optimizer.step()
            if i % 10 == 0:
                print("i:", i, "/", int(43040 / Config.train_batch_size))
            if i % 50 == 0:
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
                iteration_number += 50
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
            scheduler.step()

        #test the network after finish each epoch, to have a brief training result.
        # correct_val = 0
        # total_val = 0
        # with torch.no_grad():
        #     for img, label in test_loader:
        #         img, label = img.to(device) , label.to(device)
        #         output, _ = model(img, img)
        #         _, predicted = torch.max(output.data, 1)
        #         total_val += label.size(0)
        #         correct_val += (predicted == label).sum().item()

        # print('Accuracy of the network on the', total_val, ': %d %%' % (100 * correct_val / total_val))
        # show_plot(counter, loss_history)

def siameseTripletTraining():
    for epoch in range(0,Config.train_number_epochs):
        for i, data in enumerate(triplet_train_loader,0):
            #Produce two sets of images with the label as 0 if they're from the same file or 1 if they're different
            anchor_img, pos_img, neg_img = data
            anchor_img, pos_img, neg_img = anchor_img.to(device), pos_img.to(device), neg_img.to(device)

            # Forward pass
            anchor_output, pos_output, neg_output  = model(anchor_img, pos_img, neg_img)
            # print("Out1len", output1.size(), "Out2len", output1.size(), "labelssize", labels.size())
            loss_triplet = criterion_triplet(anchor_output, pos_output, neg_output)

            # Backward and optimize
            optimizer.zero_grad()
            loss_triplet.backward()
            optimizer.step()
            if i % 10 == 0:
                print("i:", i, "/", int(43040 / Config.train_batch_size))
            if i % 50 == 0:
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_triplet.item()))
                iteration_number += 50
                counter.append(iteration_number)
                loss_history.append(loss_triplet.item())
            scheduler.step()

siameseTripletTraining()

torch.save(model.state_dict(), "model.pt")
print("Model Saved Successfully")

# --------------
# Test the model
print("> Testing")
start = time.time() #time generation
model.eval()
correct_val = 0
total_val = 0
with torch.no_grad():
    correct = 0
    total = 0
    for img, label in test_loader:
        img, label = img.to(device) , label.to(device)
        output, _ = model(img, img)
        _, predicted = torch.max(output.data, 1)
        total_val += label.size(0)
        correct_val += (predicted == label).sum().item()

    print('Test Accuracy: {} %'.format(100 * correct / total))
end = time.time()
elapsed = end - start
print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
show_plot(counter, loss_history)
print('END')