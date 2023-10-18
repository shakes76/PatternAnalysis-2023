"""
File: train.py
Author: Maia Josang
Description: Trains the Super-Resolution model using the ADNI brain dataset and saves the trained model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from modules import SuperResolutionModel
from dataset import ADNIDataset

# Hyperparameters
learning_rate = 0.001
num_epochs = 10

# Initialize the dataset and data loaders
data_loader = ADNIDataset()
train_loader = data_loader.train_loader
target_loader = data_loader.train_target_loader
test_loader = data_loader.test_loader

# Initialize the model
model = SuperResolutionModel()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store training and testing losses
train_losses = []
test_losses = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for train_batch, target_batch in zip(train_loader, target_loader):
        train_inputs, train_targets = train_batch # input shape is: 60x60, target shape is: 240x240
        target_inputs, target_targets = target_batch # input shape is: 60x60, target shape is: 240x240
        optimizer.zero_grad()
        outputs = model(train_inputs) #outputs shape is: 240x240

        # Calculate the loss by comparing super-resolved images with target images
        loss = criterion(outputs, target_inputs) # Compare with outputs and targets
        loss.backward() # Backpropagation
        optimizer.step() # Optimizer update
        train_loss += loss.item()

    train_loss /= len(train_loader) # Calculate the average training loss for the epoch
    train_losses.append(train_loss)

    # Print training loss for this epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}")

# Plot training losses
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
torch.save(plt.show(), "super_resolution_model.pth")

# Save the trained model
torch.save(model.state_dict(), "super_resolution_model.pth")
print("Model saved!")
