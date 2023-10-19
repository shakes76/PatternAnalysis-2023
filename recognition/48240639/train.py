
"""
Created on Wednesday October 18 
Siamese Network Training Script

This script is used to train a Siamese Network model on a dataset, with support for validation.
It includes training and validation loops, model saving, and TensorBoard logging.

@author: Aniket Gupta 
@ID: s4824063

"""
from modules import SiameseNN
from dataset import get_training
import os
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from torch import nn

def train(model, dataloader, device, optimizer, epoch):
    model.train()
    criterion = nn.BCELoss()
    correct = 0
    total_samples = 0

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

def validate(model, dataloader, device):
    print("Starting Validation.")
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.BCELoss()

    with torch.no_grad():
        for (images_1, images_2, targets) in dataloader:
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()
            test_loss += criterion(outputs, targets).sum().item()
            pred = torch.where(outputs > 0.5, 1, 0)
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(dataloader.dataset)

    print('\nVal set: Average loss: {:.4f}, Validation Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dataloader.dataset),
        100. * correct / len(dataloader.dataset)))

    print("Finished Validation.")

