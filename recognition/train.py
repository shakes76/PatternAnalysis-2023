import dataset
import modules
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np

# Dice Loss Function
def dice_loss(output, target):
    epsilon = 10**-8 
    intersection = (output * target).sum()
    denominator = (output + target).sum() + epsilon
    loss = 1 - (2.0 * intersection) / (denominator)
    return loss

# Variable numList must be a list of number types only
def get_average(numList):
    """
    Calculates Averages of a list of number types.

    numList: a List object containing only number types
    return: number type (float, int, etc...)
    """
    size = len(numList)
    count = 0
    for num in numList:
        count += num
    
    return count / size

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Hyper Parameters
num_epochs = 30
initial_lr = 5e-4
batch_size = 16
lr_decay = 0.985
l2_weight_decay = 1e-5

# define model
model = modules.UNet2D(3, 1).to(device)

# Datasets & Loaders
train_dataset = dataset.ISICDataset(dataset_type='training', transform=dataset.get_transform())
val_dataset = dataset.ISICDataset(dataset_type='validation', transform=dataset.get_transform())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Optimisation & Loss Settings
optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=l2_weight_decay)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)

# Training
start_time = time.time()

train_losses = []
train_dice = []
validation_losses = []
validation_dice = []

curr_epoch = 0
for epoch in range(num_epochs):
    curr_epoch += 1
    losses = []
    coeffs = []
    running_loss = 0.0

    # training
    model.train()

    for i, data in enumerate(train_loader, 0):
        print("train: ", i)
        images, labels = data['image'].to(device), data['mask'].to(device)

        outputs = model(images)
        loss = dice_loss(outputs, labels)
        losses.append(loss.item())
        coeffs.append((1 - dice_loss(outputs, labels)).item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 10 == 9:
            print(f"loss: {running_loss / 10:.3f}")
            running_loss = 0.0
        
    scheduler.step()
    train_loss = statistics.mean(losses)
    train_coeff = statistics.mean(coeffs)

    # validation
    losses = []
    coeffs = []

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            print("validate ", i)
            images, labels = data['image'].to(device), data['mask'].to(device)

            outputs = model(images)
            loss = dice_loss(outputs, labels)
            losses.append(loss.item())
            coeffs.append((1 - dice_loss(outputs, labels)).item())

            # if (i == 0):
            #     save_segments(images, labels, outputs, 9, epochNumber)
    
    valid_loss = statistics.mean(losses)
    valid_coeff = statistics.mean(coeffs)
    # ///

    train_losses.append(train_loss)
    train_dice.append(train_coeff)
    validation_losses.append(valid_loss)
    validation_dice.append(valid_coeff)

    print ("Epoch [{}/{}], Training Loss: {:.5f}, Training Dice Similarity {:.5f}".format(epoch+1, num_epochs, train_losses[-1], train_dice[-1]))
    print('Validation Loss: {:.5f}, Validation Average Dice Similarity: {:.5f}'.format(get_average(validation_losses) ,get_average(validation_dice)))

print("Training & Validation Took " + str((time.time() - start_time)/60) + " Minutes")