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

from dataset import create_siamese_dataloader,get_transforms_training, get_transforms_testing
from modules import SiameseNetwork, EmbeddingNetwork



#--------- SET DEVICE --------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#-------- LOAD TRAINING DATA --------
ROOT_DIR_TRAIN = "/home/groups/comp3710/ADNI/AD_NC/train"
train_loader = create_siamese_dataloader(ROOT_DIR_TRAIN, batch_size=32, transform=get_transforms_training())

#---------- HYPERPARAMETERS --------
learning_rate = 0.01  # Example
num_epochs = 40

#--------- INITIATE MODEL -------
h, w = 240, 256
model = SiameseNetwork(input_shape=(1, h, w)).to(device) 

#---------- DEFINE LOSS AND OPTIMISER ------
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Every 10 epochs, the learning rate will be multiplied by 0.1.
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.BCELoss()  # Using binary cross-entropy loss


#---------- TRAINING LOOP ----------
print("---------Training---------")

losses = []
accuracies = []

for epoch in range(num_epochs):

    running_loss = 0.0
    correct_pairs = 0
    total_pairs = 0

    for img1, img2, labels in train_loader:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(img1, img2)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # For accuracy
        predicted = (outputs > 0.5).float()
        correct_pairs += (predicted.squeeze() == labels).sum().item()
        total_pairs += labels.size(0)
    
    scheduler.step()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct_pairs / total_pairs

    losses.append(epoch_loss)
    accuracies.append(epoch_acc)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

print("Training complete")

# Save Model
print("Saved model after training")
torch.save(model.state_dict(), 'model_40_epoch_augment_lrscheduler.pth')


# Plot the accuracy and loss
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(losses, 'g-')
ax2.plot(accuracies, 'b-')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss', color='g')
ax2.set_ylabel('Training Accuracy', color='b')
plt.savefig('training_plot.png')

#------------------- TESTING -------------------------
print("------Testing---------")
ROOT_DIR_TEST = "/home/groups/comp3710/ADNI/AD_NC/test"
test_loader = create_siamese_dataloader(ROOT_DIR_TEST, batch_size=32, transform=get_transforms_testing())

model.eval()

correct_pairs = 0
total_pairs = 0


with torch.no_grad():
    for img1, img2, labels in test_loader:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        outputs = model(img1, img2)
        predicted = (outputs > 0.5).float().squeeze()  # Ensure squeezing if necessary
        correct_pairs += (predicted == labels).sum().item()
        total_pairs += labels.size(0)

# 4. Compute metrics
accuracy = 100 * correct_pairs / total_pairs
print(f"Accuracy on test data: {accuracy:.2f}%")

