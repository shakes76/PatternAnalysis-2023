import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import train_dataset, test_dataset
from modules import SiameseResNet

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create data loaders for training and testing datasets
train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=8, batch_size=8)
test_dataloader = DataLoader(test_dataset, shuffle=True, num_workers=8, batch_size=1)

# Initialize tensorboard writer
writer = SummaryWriter()

# Initialize the Siamese network and move it to the appropriate device
net = SiameseResNet().to(device)

# Define the loss function
criterion = nn.BCEWithLogitsLoss()

# Define the optimizer
optimizer = optim.Adam(net.parameters(), lr=0.0005)

# Set the network to training mode
net.train()

# Initialize the best loss to a very large number
best_loss = float('inf')
# Number of epochs to wait for improvement
patience = 5
# Number of epochs with no improvement after which training will be stopped
no_improve_epochs = 0

# Training loop with early stopping
for epoch in range(0, 50):
    # Initialize the running loss to 0
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # Get the images and labels from the data
        img0, img1, label = data

        # Move the images and labels to the appropriate device
        img0, img1, label = img0.to(device), img1.to(device), label.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass: Compute predicted outputs by passing inputs to the model
        output1, output2 = net(img0, img1)

        # Calculate the loss
        loss = criterion(output1, output2, label)

        # Backward pass: Compute gradient of the loss with respect to model parameters
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Write the loss value to tensorboard
        running_loss += loss.item()

    # Calculate the average loss over the entire training data
    avg_loss = running_loss / len(train_dataloader)
    # Print the average loss value
    print("Epoch: {}, Loss: {}".format(epoch, avg_loss))
    # Write the average loss value to tensorboard
    writer.add_scalar('Training Loss', avg_loss, epoch)

    # Early stopping
    # If the current loss is less than the best loss, set the best loss to the current loss
    if avg_loss < best_loss:
        # Save the model parameters
        best_loss = avg_loss
        # Reset the number of epochs with no improvement
        no_improve_epochs = 0
    else:
        # Otherwise increment the number of epochs with no improvement
        no_improve_epochs += 1

    # If the number of epochs with no improvement has reached the patience limit, stop training
    if no_improve_epochs >= patience:
        print("Early stopping!")
        break

# Save the trained model
torch.save(net.state_dict(), "model.pth")
