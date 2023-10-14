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
Siamese_A = SiameseNetwork()
Siamese_B = SiameseNetwork()

Siamese_A.to(device)  # Move the network to GPU if available
Siamese_B.to(device)  # Move the network to GPU if available


# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss

optimizer = optim.Adam([
    {'params': Siamese_A.parameters()},
    {'params': Siamese_B.parameters()}
], lr=1e-3)



loss.backward()
optimizer.step()

# Training loop
num_epochs = 10



for epoch in range(num_epochs):
    running_loss = 0.0
    for batch in iter(trainloader):
        images, labels = batch
        
        images = images.to(device)
        labels = labels.to(device)

        # Backwards and optimise
        optimizer.zero_grad()

        # NEED TO TAKE IN TWO SEPARATE IMAGES
        

        # CALCULATE L1 OR PAIRWISE DISTANCE

        # CALCULATE CONTRASTIVE LOSS


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

# step 1: train the siamese + save the model