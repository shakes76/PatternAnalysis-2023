
"""
Created on Wednesday October 18 
Siamese Network Training Script

This script is used to train a Siamese Network model on a dataset, with support for validation.
It includes training and validation loops, model saving, and TensorBoard logging.

@author: Aniket Gupta 
@ID: s4824063

"""
#Import necessary libraries 
from modules import SiameseNN
from dataset import get_training
import os
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from torch import nn
# Define the training function
def training(model, dataloader, device, optimizer, epoch):
    model.train()
    total_samples = 0
    correct = 0
    criterion = nn.BCELoss()

    for batch_idx, (images_1, images_2, targets) in enumerate(dataloader):
        images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images_1, images_2).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        pred = torch.where(outputs > 0.5, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
        correct += pred.eq(targets.view_as(pred)).sum().item()
        total_samples += len(targets)

        if batch_idx % 10 == 0:
            accuracy = 100. * correct / total_samples

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}%'.format(
                epoch, total_samples, len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item(), accuracy))

            writer.add_scalar('Training Loss', loss.item(), epoch * len(dataloader) + batch_idx)
            writer.add_scalar('Training Accuracy', accuracy, epoch * len(dataloader) + batch_idx)
# Define the validation function
def check(model, dataloader, device):
    print("Starting Checks.")
    model.eval()
    criterion = nn.BCELoss()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for (images_1, images_2, targets) in dataloader:
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()
            test_loss += criterion(outputs, targets).sum().item()
            pred = torch.where(outputs > 0.5, 1, 0)
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(dataloader.dataset)

    print('\nVal set: Centralized  loss: {:.4f}, Validate Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dataloader.dataset),
        100. * correct / len(dataloader.dataset)))

    print("Finished Checked.")

if __name__ == '__main__':
    writer = SummaryWriter('logs/siam_net_exp')
    epochs = 50
    batch_size = 2816
    learning_rate = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = SiameseNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Loading data...")
    train_data = get_training('/Users/aniketgupta/Desktop/Pattern Recognition/PatternAnalysis-2023/recognition/48240639/AD_NC')
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    print("Data loaded.")

    save_directory = "/Users/aniketgupta/Desktop/Pattern Recognition/PatternAnalysis-2023/results"
    save_filename = f"siam_net_{epochs}epochs.pt"

    print("Training Started.")
    for epoch in range(1, epochs + 1):
        training(model, train_dataloader, device, optimizer, epoch)
    print("Finished training.")

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    save_path = os.path.join(save_directory, save_filename)
    torch.save(model.state_dict(), save_path)

    writer.close()
