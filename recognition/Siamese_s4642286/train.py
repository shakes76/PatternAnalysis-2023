"""
Name: train.py
Student: Ethan Pinto (s4642286)
Description: Containing the source code for training, validating, testing and saving your model.
"""

import torch
import torch.optim as optim
import torch.nn as nn
from modules import SiameseNetwork
from dataset import trainloader, testloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print('CUDA not available. Running on CPU...')

print(device)

# Initialize the Siamese network
SNN = SiameseNetwork()
SNN.to(device)  # Move the network to GPU if available

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(SNN.parameters(), lr=0.001)  # Example optimizer (adjust learning rate)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = [img.to(device) for img in inputs]  # Move inputs to GPU

        optimizer.zero_grad()

        outputs = SNN(*inputs)  # Forward pass
        loss = criterion(outputs, labels.float().to(device))  # Calculate loss

        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()

    # Print the average loss for this epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(trainloader)}")

print("Finished Training")


# Save the model's state dictionary
torch.save(SNN.state_dict(), "./model")