"""
train.py

Code for training and saving the model.

Author: Atharva Gupta
Date Created: 17-10-2023
"""
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

from datasplit import create_data_loader
from modules import CustomPerceiver
from parameter import *

# Create a data loader for training data
train_data_loader = create_data_loader(DATA_PATH, BATCH_SIZE)

# Check if a GPU is available, if not, use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the custom Perceiver model
model = CustomPerceiver(NUM_LATENTS, DIM_LATENTS, DEPTH_LATENT_TRANSFORMER, NUM_CROSS_ATTENDS)
model.to(device)

# Define loss function (cross-entropy) and optimizer (Adam)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

loss_data = []  # List to store average loss for each epoch
accuracy = []   # List to store training accuracy for each epoch

for epoch in range(EPOCHS):
    correct = 0
    total = 0
    running_loss = 0.0

    for i, data in enumerate(train_data_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()

    print(f"Epoch {epoch} completed")
    loss_data.append(running_loss / len(train_data_loader))
    accuracy.append(correct / total * 100)

# Plot average loss over epochs
plt.plot(loss_data)
plt.xlabel('EPOCH')
plt.ylabel('Average Loss')
plt.show()

# Plot training accuracy over epochs
plt.plot(accuracy)
plt.xlabel('EPOCH')
plt.ylabel('Training Accuracy')
plt.show()

# Save the trained model's state dictionary to the specified path
torch.save(model.state_dict(), MODEL_PATH)
