import torch.nn as nn
import os
import numpy as np
import random
from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from PIL import Image
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

from dataset import create_siamese_dataloader,get_transforms_training, get_transforms_testing
from modules import SiameseResNet, ContrastiveLoss, ClassifierNet






#--------- SET DEVICE --------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#-------- LOAD TRAINING DATA --------
ROOT_DIR_TRAIN = "/home/groups/comp3710/ADNI/AD_NC/train"
train_loader, val_loader = create_siamese_dataloader(ROOT_DIR_TRAIN, batch_size=32, transform=get_transforms_training(), split_flag=True)

#---------- HYPERPARAMETERS --------
learning_rate =  0.01 
num_epochs = 40
margin = 1


#--------- INITIATE MODEL -------
model = SiameseResNet().to(device)

#---------- DEFINE LOSS AND OPTIMISER ------
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = ContrastiveLoss(margin=margin)


#---------- TRAINING LOOP ----------
print("---------Training Siamese Network---------")

import matplotlib.pyplot as plt

# Initialize lists to store loss values
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0

    for batch_idx, (img1, img2, labels, _, _) in enumerate(train_loader):
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

        optimizer.zero_grad()

        distances = model(img1, img2)
        loss = criterion(distances, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)  # Store the training loss

    # Validation
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
         for batch_idx, (img1, img2, labels, _, _) in enumerate(val_loader):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            distances = model(img1, img2)
            loss = criterion(distances, labels)
            
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)  # Store the validation loss

    print('[Epoch %d] training loss: %.3f, validation loss: %.3f' % (epoch + 1, avg_train_loss, avg_val_loss))

print("Training complete")

# Save Model
print("Saved siamese after training")
torch.save(model.state_dict(), 'siamese_40.pth')

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss", color="blue")
plt.plot(val_losses, label="Validation Loss", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot as an image
plt.savefig("loss_curve.png")






# #------------------- TESTING -------------------------
# print("------Testing---------")
# ROOT_DIR_TEST = "/home/groups/comp3710/ADNI/AD_NC/test"
# test_loader = create_siamese_dataloader(ROOT_DIR_TEST, batch_size=32, transform=get_transforms_testing(),split_flag = False)


# model.eval()

# correct = 0
# total = 0

# with torch.no_grad():  # No need to compute gradients during evaluation
#     for img1, img2, labels,_,_ in test_loader:
#         img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

#         outputs = model(img1, img2).squeeze()
#         preds = (outputs >= 0.5).float().squeeze()
        
#         total += labels.size(0)
#         correct += (preds == labels).sum().item()

# accuracy = 100 * correct / total
# print(f"Accuracy on test data: {accuracy:.2f}%")


# # Save the training and validation loss plot
# plt.figure(figsize=(10, 5))
# plt.plot(train_loss_list, label='Training Loss')
# plt.plot(val_loss_list, label='Validation Loss')
# plt.title('Training and Validation Loss over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig('training_validation_loss_90_nonorm_withinitweights.png')  # Save the figure
# plt.close()  # Close the figure to prevent it from being displayed

# # Save the training and validation accuracy plot
# plt.figure(figsize=(10, 5))
# plt.plot(train_accuracy_list, label='Training Accuracy')
# plt.plot(val_accuracy_list, label='Validation Accuracy')
# plt.title('Training and Validation Accuracy over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.savefig('training_validation_accuracy_90_nonorm_withinitweights.png')  # Save the figure
# plt.close() 
# print("Plots saved")

