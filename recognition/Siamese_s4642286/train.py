"""
Name: train.py
Student: Ethan Pinto (s4642286)
Description: Containing the source code for training, validating, testing and saving your model.
"""

import torch
import torch.optim as optim
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from modules import SiameseNetwork, ContrastiveLoss, MLP
from dataset import trainloader, mlp_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print('CUDA not available. Running on CPU...')

print(device)


##########################################################################################
################                  Training Siamese Network                ################
##########################################################################################

siamese_net = SiameseNetwork()
siamese_net.to(device)

# Define loss function and optimizer
criterion = ContrastiveLoss(margin=2.0)
optimizer = optim.Adam(siamese_net.parameters(), lr=1e-3)

# Training loop
num_epochs = 10
loss_history = []

print("Starting Training Siamese Network...")
start = time.time()
siamese_net.train()

for epoch in range(num_epochs):
    total_loss = 0.0  # Track the total loss for the epoch

    for batch_idx, batch in enumerate(trainloader, 0):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Pass each image individually through the Siamese Network
        output1, output2 = siamese_net(inputs[:, 0], inputs[:, 1])

        # Calculate the loss for the batch and backpropagate
        loss = criterion(output1, output2, labels)
        loss.backward()

        optimizer.step()

        # Accumulate the loss for the epoch
        total_loss += loss.item()

    # Print the average loss for the epoch
    avg_loss = total_loss / len(trainloader)  # Calculate the average loss for the epoch
    loss_history.append(avg_loss)

    print(f"Epoch {epoch + 1}/{num_epochs} -> Average loss: {avg_loss}")

print("Finished Training Siamese Network...")

# Display and Save Figure
plt.figure(figsize=(10,5))
plt.title("Siamese Network Training Loss")
plt.plot(range(1, num_epochs + 1), loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
plt.savefig('siamese_loss_plot1.png', dpi=300)


end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")


# Save the trained model
torch.save(siamese_net.state_dict(), "PatternAnalysis-2023/recognition/Siamese_s4642286/Siamese/siamese_model.pt")


# ##########################################################################################
# ################                      Training MLP                          ##############
# ##########################################################################################

siamese = SiameseNetwork()
siamese.to(device)
siamese.train()

# load in saved parameters
siamese.load_state_dict(torch.load("PatternAnalysis-2023/recognition/Siamese_s4642286/Siamese/siamese_model.pt"))

# Define the input size, hidden size, and output size
input_size = 128
hidden_size = 64 
output_size = 1  # Output size is 1 for binary classification

# Create an instance of the SimpleMLP model
mlp = MLP(input_size, hidden_size, output_size)
mlp.to(device)
mlp.train()

# Define binary cross-entropy loss and optimizer for MLP
criterion = nn.BCELoss()
optimizer = optim.SGD(mlp.parameters(), lr=1e-3)

# Training loop for MLP
num_epochs = 10
epoch_losses = []  # List to store the loss for each epoch

print("Starting Training MLP...")
start = time.time()

for epoch in range(num_epochs):
    iteration_number = 0
    loss_history = []

    for batch_idx, (input, label) in enumerate(mlp_loader):
        input, label = input.to(device), label.to(device)

        optimizer.zero_grad()

        # Images -> Siamese -> MLP -> Output
        input = siamese.forward_once(input)
        output = mlp(input)

        # Calculate Binary Cross Entropy Loss
        loss = criterion(output, label.view(-1, 1).float())
        loss.backward()

        optimizer.step()

        loss_history.append(loss.item())

        if batch_idx % 10 == 0:
            iteration_number += 10

    # Calculate the average loss for the epoch
    epoch_loss = sum(loss_history) / len(loss_history)
    epoch_losses.append(epoch_loss)

    print(f"Epoch number: {epoch} -> Average loss: {epoch_loss:.4f}")

print("Finished Training MLP...")

# Display and Save Figure
plt.figure(figsize=(10, 5))
plt.title("MLP Training Loss")
plt.plot(range(1, num_epochs + 1), epoch_losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
plt.savefig('MLP_loss_plot1.png', dpi=300)

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

# Save the trained MLP
torch.save(mlp.state_dict(), "PatternAnalysis-2023/recognition/Siamese_s4642286/Classifier/classifier.pt")