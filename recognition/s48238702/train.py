"""
train.py: Train a Siamese Network for image similarity learning.

Author: Rachit Chaurasia (s4823870)
Date: 20/10/2023

This script trains a Siamese Network for image similarity learning. The trained Siamese Network
model is saved to a file specified by `SNN_PATH`. Additionally, the script plots the training loss.

The Siamese Network is trained using the `trainSNN` function, which trains the network for a specified
number of epochs. The network is trained using the provided data and saves the trained model to the
specified file.

The script uses a custom Siamese dataset and the Contrastive Loss for training.

To run the training, execute this script in your preferred environment.

"""

import torch
import torch.optim as optim
import torchvision
import tqdm
import matplotlib.pyplot as plt
from dataset import load_siamese_data
from modules import SiameseNetwork, ContrastiveLoss

SNN_PATH = 'SNN.pth'  # Path to save the trained Siamese Network model

def train():
    """
    Train the Siamese Network, save the model, and plot the loss.
    """
    # Train and Save the Siamese Network
    siamese_fit = trainSNN()
    torch.save(siamese_fit['model'].state_dict(), SNN_PATH)

    # Plot Accuracy and Loss
    plot_data(siamese_fit['loss'], 'Siamese Network')
    plt.show()

def trainSNN(epochs=50):
    """
    Train the Siamese Network.
    
    Args:
        epochs (int, optional): Number of training epochs. Default is 30.
    
    Returns:
        dict: A dictionary containing the trained model and loss history.
    """
    siamese_train_loader = load_siamese_data(batch_size=32)
    model = SiameseNetwork()
    
    # Check if CUDA (GPU) is available and move the model to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    siamese_fit = {'model': model, 'loss': {'train': [], 'val': []}}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # Wrap the data loader with tqdm to add a progress bar
        with tqdm.tqdm(siamese_train_loader, unit="batch") as data_loader:
            for batch_idx, (img1, img2, labels) in enumerate(data_loader):
                optimizer.zero_grad()
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                output1, output2 = model(img1, img2)
                loss = criterion(output1, output2, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                # Update the progress bar description
                data_loader.set_description(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / (batch_idx + 1):.4f}")
        
        siamese_fit['loss']['train'].append(train_loss / len(siamese_train_loader))
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {siamese_fit["loss"]["train"][-1]}')
    
    return siamese_fit

def plot_data(data, title):
    """
    Plot training data (e.g., loss).

    Args:
        data (dict): Data to be plotted, typically containing training and validation loss.
        title (str): Title for the plot.
    """
    plt.figure()
    plt.plot(data['train'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title)

if __name__ == '__main__':
    train()
