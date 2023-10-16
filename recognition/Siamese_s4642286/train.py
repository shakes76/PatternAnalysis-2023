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
from dataset import trainloader, mlp_trainloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print('CUDA not available. Running on CPU...')

print(device)


##########################################################################################
################                  Training Siamese Network                  ##############
##########################################################################################

# Initialize the Siamese network
# CNN_A = CNN()
# CNN_B = CNN()

# # Move the network to GPU if available
# CNN_A.to(device)
# CNN_B.to(device)  

# Siamese = SiameseNetwork(CNN_A, CNN_B)
# Siamese.to(device)


siamese_net = SiameseNetwork()
siamese_net.to(device)


# Define loss function and optimizer
criterion = ContrastiveLoss(margin=2.0)
optimizer = optim.Adam(siamese_net.parameters(), lr=1e-3)

# Training loop
num_epochs = 10
counter = []
loss_history = [] 
iteration_number= 0

print("Starting Training Siamese Network...")
start = time.time()
siamese_net.train()

for epoch in range(num_epochs):
    total_loss = 0.0  # Track the total loss for the epoch

    for batch_idx, batch in enumerate(trainloader, 0):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        output1, output2 = siamese_net(inputs[:, 0], inputs[:, 1])

        loss = criterion(output1, output2, labels)
        loss.backward()

        optimizer.step()
        
        # Accumulate the loss for the epoch
        total_loss += loss.item()

    # Print the average loss for the epoch
    avg_loss = total_loss / len(trainloader)  # Calculate the average loss for the epoch
    print(f"Epoch number: {epoch} -> Average loss: {avg_loss}")
    counter.append(iteration_number)
    loss_history.append(avg_loss)
    iteration_number += 1


print("Finished Training Siamese Network...")

# Display and Save Figure
plt.figure(figsize=(10,5))
plt.title("Siamese Network Training Loss")
plt.plot(counter, loss_history)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig('siamese_loss_plot.png', dpi=300)

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")


# Save the trained model
torch.save(siamese_net.state_dict(), "PatternAnalysis-2023/recognition/Siamese_s4642286/Siamese/siamese_model.pt")


# ##########################################################################################
# ################                    Training MLP                            ##############
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
counter = []
loss_history = [] 
iteration_number= 0

print("Starting Training MLP...")
start = time.time()

for epoch in range(num_epochs):

    for batch_idx, (input, label) in enumerate(mlp_trainloader):
        input, label = input.to(device), label.to(device)

        optimizer.zero_grad()

        # Images -> Siamese -> MLP -> Output
        input = siamese.forward_once(input)
        output = mlp(input)

        # Calculate Binary Cross Entropy Loss
        loss = criterion(output, label.view(-1, 1).float())
        loss.backward()

        optimizer.step()

        # Every 10 batches print out the loss
        if batch_idx % 10 == 0 :
            print(f"Epoch number: {epoch} -> Current loss: {loss.item()}\n")
            iteration_number += 10

            counter.append(iteration_number)
            loss_history.append(loss.item())


print("Finished Training MLP...")

# Display and Save Figure
plt.figure(figsize=(10,5))
plt.title("MLP Training Loss")
plt.plot(counter, loss_history)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig('MLP_loss_plot.png', dpi=300)

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

# Save the trained MLP
torch.save(mlp.state_dict(), "PatternAnalysis-2023/recognition/Siamese_s4642286/Classifier/classifier.pt")