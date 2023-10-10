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
import os

# Dice Loss Function
def dice_loss(output, target):
    epsilon = 10**-8 
    intersection = (output * target).sum()
    denominator = (output + target).sum() + epsilon
    loss = 1 - (2.0 * intersection) / (denominator)
    return loss

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Hyper Parameters
num_epochs = 1
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

            # plots
            if not os.path.exists('plots'):
                os.makedirs('plots')

            fig, axs = plt.subplots(5, 3, figsize=(15,5*5))
            axs[0][0].set_title("Original Image")
            axs[0][1].set_title("Actual Mask")
            axs[0][2].set_title("Predicted Mask")
            for row in range(5):
                img = images.cpu()[row].permute(1,2,0).numpy()
                label = labels.cpu()[row].permute(1,2,0).numpy()
                pred = outputs.cpu()[row]
                pred = pred.permute(1,2,0).numpy()
                axs[row][0].imshow(img)
                axs[row][0].xaxis.set_visible(False)
                axs[row][0].yaxis.set_visible(False)

                axs[row][1].imshow(label, cmap="gray")
                axs[row][1].xaxis.set_visible(False)
                axs[row][1].yaxis.set_visible(False)

                axs[row][2].imshow(pred, cmap="gray")
                axs[row][2].xaxis.set_visible(False)
                axs[row][2].yaxis.set_visible(False)
            
                fig.suptitle("Validation Segments Epoch: " + str(curr_epoch))
                plt.savefig(os.path.join('plots', "ValidationSegmentsEpoch" + str(curr_epoch)))

            plt.close()
    
    valid_loss = statistics.mean(losses)
    valid_coeff = statistics.mean(coeffs)
    # ///

    train_losses.append(train_loss)
    train_dice.append(train_coeff)
    validation_losses.append(valid_loss)
    validation_dice.append(valid_coeff)

    print ("epoch_num = {}/{}, Training Loss: {:.5f}, Training Dice Similarity {:.5f}".format(epoch+1, num_epochs, train_losses[-1], train_dice[-1]))
    print('Val Loss: {:.5f}, Val Average Dice Similarity: {:.5f}'.format(statistics.mean(validation_losses), statistics.mean(validation_dice)))

print("Total Time: " + str((time.time() - start_time)/60) + " Minutes")
    
x = list(range(1, len(train_losses) + 1))

plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.plot(x, train_losses, label=f"Training Loss")
plt.plot(x, validation_losses, label=f"Validation Loss")
plt.legend()

plt.title(f"Training and Validation Loss Over Epochs")
plt.savefig(os.path.join('plots', "results"))
plt.close()