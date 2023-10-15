"""
Filename: train.py
Author: Benjamin Guy
Date: 15/10/2023
Description: This file contains code required to train the ESPCN model on the ADNI dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from modules import ESPCN
from dataset import get_train_and_validation_loaders, get_test_loader
import time
import matplotlib.pyplot as plt

MODEL_PATH = "recognition\\46972990 - MRI super-resolution network\\model.pth"

# Create the model and load training data
model = ESPCN(upscale_factor=4, channels=1)
train_loader, validation_loader = get_train_and_validation_loaders()
test_loader = get_test_loader()

# Move the model onto the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(torch.cuda.get_device_name(torch.cuda.current_device()))
model.to(device)

# Set training parameters
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

# Store losses for visualisation
train_losses = []
val_losses = []

print("Started training...")
for epoch in range(num_epochs):
    start_time = time.time()

    # Training Loop
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}", end=" - ")
    
    # Validation Loop
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for i, data in enumerate(validation_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss = val_loss / len(validation_loader)
    print(f"Validation Loss: {val_loss:.4f}", end=" - ")

    # Time taken for epoch
    end_time = time.time()
    epoch_duration = end_time - start_time
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Completed in {epoch_duration:.2f} seconds.")

print("Finished training.")

# Testing the model
def evaluate_model(model, test_loader, device, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(test_loader)

# Output testing loss
test_loss = evaluate_model(model, test_loader, device, criterion)
print(f"Test Loss: {test_loss:.4f}")

# Save the final model
torch.save(model.state_dict(), MODEL_PATH)

# Plot the training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training & Validation Losses Over Epochs")
plt.legend()
plt.show()