"""
Name: train.py
Student: Ethan Pinto (s4642286)
Description: Containing the source code for training, validating, testing and saving your model.
"""

import torch
import torch.optim as optim
import torch.nn as nn
import time
from modules import SiameseNetwork, ContrastiveLoss, CNN, MLP
from dataset import trainloader

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

print("Starting Training...")
start = time.time()
Siamese.train()

for epoch in range(num_epochs):
    total_loss = 0.0

    for batch_idx, batch in enumerate(trainloader):
        inputs, labels = batch

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        
        output1, output2 = Siamese(inputs[:, 0], inputs[:, 1])
        
        loss = criterion(output1, output2, labels)
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss.item()

    # Print the average loss for the epoch
    print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss / (batch_idx + 1):.4f}')

print("Finished Training...")
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

# Save the trained model
torch.save(Siamese.state_dict(), "./Siamese/siamese_model.pt")


# Instantiate the MLP and move it to the GPU if available
cnn = CNN()  # Assuming you have a CNN instance
cnn.to(device)

# load in saved parameters
cnn.load_state_dict(torch.load("./Siamese/siamese_model.pt"))

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

    for batch_idx, (input, label) in enumerate(trainloader):  # Replace 'dataloader' with your data loading logic
        input, label = input.to(device), label.to(device)

        optimizer.zero_grad()

        input = cnn(input)

        output = mlp(input)
        loss = criterion(output, label.view(-1, 1).float())  # Ensure label is a tensor of shape (batch_size, 1)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    # Print the average loss for the epoch
    print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss / (batch_idx + 1):.4f}')

# Save the trained MLP
torch.save(mlp.state_dict(), "./Classifier")