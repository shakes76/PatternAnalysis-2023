"""
Created on Wednesday October 18 
Siamese Network Training Script

This script is used to train a Siamese Network model on a dataset, with support for validation.
It includes training and validation loops, model saving, and TensorBoard logging.

@author: Aniket Gupta 
@ID: s4824063

"""

# Import necessary libraries
import os  # Import the os module for file operations
import torch  # Import the PyTorch library
import torch.optim as optim  # Import the optimizer module from PyTorch
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard for logging
from torch.utils.data import DataLoader  # Import DataLoader for handling data batches
from modules import SiameseNN  # Import the Siamese Network model
from dataset import get_training  # Import the function for loading the training dataset
from torch import nn  # Import the neural network module from PyTorch

# Training function
def train(model, dataloader, device, optimizer, epoch):
    """
    Training function for the Siamese Network.

    Args:
        model (nn.Module): The Siamese Network model.
        dataloader (DataLoader): DataLoader for training data.
        device (torch.device): The device to perform training on (CPU or GPU).
        optimizer (Optimizer): The optimizer used for training.
        epoch (int): The current training epoch.

    Returns:
        None
    """
    model.train()  # Set the model to training mode
    total_samples, correct, criterion = 0, 0, nn.BCELoss()  # Initialize variables

    for batch_idx, (images_1, images_2, targets) in enumerate(dataloader):
        images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images_1, images_2).squeeze()  # Forward pass to obtain model's predictions
        loss = criterion(outputs, targets)  # Calculate the loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimizer step

        pred = (outputs > 0.5).float()  # Convert model outputs to binary predictions
        correct += pred.eq(targets.view_as(pred)).sum().item()  # Count correct predictions
        total_samples += len(targets)

        if batch_idx % 10 == 0:
            accuracy = 100. * correct / total_samples
            print(f'Train Epoch: {epoch} [{total_samples}/{len(dataloader.dataset)} '
                  f'({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}\tAccuracy: {accuracy:.2f}%')

            writer.add_scalar('Training Loss', loss.item(), epoch * len(dataloader) + batch_idx)
            writer.add_scalar('Training Accuracy', accuracy, epoch * len(dataloader) + batch_idx)

# Validation function
def validate(model, dataloader, device):
    """
    Validation function for the Siamese Network.

    Args:
        model (nn.Module): The Siamese Network model.
        dataloader (DataLoader): DataLoader for validation data.
        device (torch.device): The device to perform validation on (CPU or GPU).

    Returns:
        None
    """
    print("Starting Validation.")
    model.eval()  # Set the model to evaluation mode (no gradient computation)
    criterion = nn.BCELoss()  # Define the loss function
    correct, test_loss = 0, 0  # Initialize variables for tracking correctness and loss

    with torch.no_grad():
        for images_1, images_2, targets in dataloader:
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()  # Forward pass for validation data
            test_loss += criterion(outputs, targets).sum().item()  # Calculate the loss
            pred = (outputs > 0.5).int()  # Convert model outputs to binary predictions
            correct += pred.eq(targets.view_as(pred)).sum().item()  # Count correct predictions

    test_loss /= len(dataloader.dataset)  # Calculate the average test loss

    print(f'\nVal set: Centralized loss: {test_loss:.4f}, Validate Accuracy: {correct}/{len(dataloader.dataset)} '
          f'({100. * correct / len(dataloader.dataset):.0f}%)\n')
    print("Finished Validation.")

if __name__ == '__main__':
    writer = SummaryWriter('logs/siam_net_exp')  # Create a TensorBoard writer
    epochs, batch_size, learning_rate = 50, 256, 0.002 # Define training epochs, batch size, and learning rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check and set the device
    print(device)
    model = SiameseNN().to(device)  # Initialize the Siamese Network model
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Initialize the optimizer

    print("Loading data...")
    train_data = get_training('/Users/aniketgupta/Desktop/Pattern Recognition/PatternAnalysis-2023/recognition/48240639/AD_NC')
    train_dataloader = DataLoader(train_data, batch_size=batch_size)  # Load training data
    print("Data loaded.")

    save_directory = "/Users/aniketgupta/Desktop/Pattern Recognition/PatternAnalysis-2023/results"
    save_filename = f"siam_net_{epochs}epochs.pt"

    print("Training Started.")
    for epoch in range(1, epochs + 1):
        train(model, train_dataloader, device, optimizer, epoch)  # Training loop
    print("Finished training.")

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    save_path = os.path.join(save_directory, save_filename)
    torch.save(model.state_dict(), save_path)  # Save the trained model

    writer.close()  # Close the TensorBoard writer
