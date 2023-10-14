"""
Name: train.py
Student: Ethan Pinto (s4642286)
Description: Containing the source code for training, validating, testing and saving your model.
"""

import torch
import torch.optim as optim
import torch.nn as nn
from modules import SiameseNetwork, ContrastiveLoss, CNN, MLP
from dataset import trainloader, testloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print('CUDA not available. Running on CPU...')

print(device)

# Initialize the Siamese network
CNN_A = CNN()
CNN_B = CNN()

# Move the network to GPU if available
CNN_A.to(device)
CNN_B.to(device)  

# Define loss function and optimizer
criterion = ContrastiveLoss(margin=2.0)

optimizer = optim.Adam([
    {'params': CNN_A.parameters()},
    {'params': CNN_B.parameters()}
], lr=1e-3)


Siamese = SiameseNetwork(CNN_A, CNN_B)
Siamese.to(device)

# Define loss function and optimizer
criterion = ContrastiveLoss(margin=2.0)
optimizer = optim.Adam([
    {'params': CNN_A.parameters()},
    {'params': CNN_B.parameters()}
], lr=1e-3)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    print("Epoch：", epoch, " start.")

    Siamese.train()
    total_loss = 0.0

    for batch_idx, (input1, input2, label) in enumerate(trainloader):
        input1, input2, label = input1.to(device), input2.to(device), label.to(device)
        
        optimizer.zero_grad()
        
        output1, output2 = Siamese(input1, input2)
        
        loss = criterion(output1, output2, label)
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss.item()

    # Print the average loss for the epoch
    print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss / (batch_idx + 1):.4f}')

print("Finished Training")

# Save the trained model
torch.save(Siamese.state_dict(), "./Siamese")



# MLP SHOULD BE TRAINED ON THE FEATURE VECTOR.
# Instantiate the MLP and move it to the GPU if available
cnn = CNN()  # Assuming you have a CNN instance
cnn.to(device)

mlp = MLP()
mlp.to(device)

# Define binary cross-entropy loss and optimizer for MLP
criterion = nn.BCELoss()
optimizer = optim.Adam(mlp.parameters(), lr=1e-3)

# Training loop for MLP
num_epochs = 10

for epoch in range(num_epochs):
    mlp.train()
    total_loss = 0.0

    for batch_idx, (input, label) in enumerate(dataloader):  # Replace 'dataloader' with your data loading logic
        input, label = input.to(device), label.to(device)

        optimizer.zero_grad()

        output = mlp(input)
        loss = criterion(output, label.view(-1, 1).float())  # Ensure label is a tensor of shape (batch_size, 1)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    # Print the average loss for the epoch
    print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss / (batch_idx + 1):.4f}')

# Save the trained MLP
torch.save(mlp.state_dict(), "./Classifier")