"""
    File name: modules.py
    Author: Fanhao Zeng
    Date created: 11/10/2023
    Date last modified: 16/10/2023
    Python Version: 3.10.12
"""

import os
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Subset
from modules import SiameseNetwork, ADNIClassifier
from dataset import get_train_dataset, get_classifier_train_dataset


def train(model, dataloader, device, optimizer, epoch, writer):
    model.train()

    # Using `BCELoss` as the loss function
    criterion = nn.BCELoss()

    correct = 0  # Reset correct count for each epoch
    total_samples = 0  # Keep track of total samples processed

    for batch_idx, (images_1, images_2, targets) in enumerate(dataloader):
        images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
        optimizer.zero_grad()  # Zero out gradients
        outputs = model(images_1, images_2).squeeze()
        loss = criterion(outputs, targets)  # Calculate loss
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        pred = torch.where(outputs > 0.5, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
        correct += pred.eq(targets.view_as(pred)).sum().item()
        total_samples += len(targets)  # Update total samples

        if batch_idx % 10 == 0:
            accuracy = 100. * correct / total_samples  # Calculate accuracy

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}%'.format(
                epoch, total_samples, len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item(), accuracy))

            # Log to TensorBoard
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
            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
            pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(dataloader.dataset)

    print('\nVal set: Average loss: {:.4f}, Validation Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dataloader.dataset),
        100. * correct / len(dataloader.dataset)))

    print("Finished Validation.")


def train_siamese(device, epochs=50, batch_size=256, learning_rate=0.001):
    # Initialize TensorBoard
    writer = SummaryWriter('logs/siamese_experiment')

    print(device)

    # Initialize model, criterion, and optimizer
    model = SiameseNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Get dataloaders
    print("Loading data...")
    train_data = get_train_dataset('E:/comp3710/AD_NC')
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    print("Data loaded.")

    save_directory = "E:/PatternAnalysis-2023/results"
    save_filename = f"siamese_network_{epochs}epochs.pt"

    print("Starting training.")
    for epoch in range(1, epochs + 1):
        train(model, train_dataloader, device, optimizer, epoch, writer)
        # validate(model, val_loader, device)
    print("Finished training.")

    # create directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # save model
    save_path = os.path.join(save_directory, save_filename)
    torch.save(model.state_dict(), save_path)

    writer.close()


def train_classifier(device, epochs=50, batch_size=256, learning_rate=0.001):
    # Initialize TensorBoard writer
    writer = SummaryWriter('logs/classifier_experiment')

    print(device)

    siamese_model = SiameseNetwork()
    model_path = "../results/siamese_network_50epochs.pt"
    siamese_model.load_state_dict(torch.load(model_path))
    siamese_model = siamese_model.to(device)

    # freeze the parameters of the Siamese Network
    for param in siamese_model.parameters():
        param.requires_grad = False

    classifier_model = ADNIClassifier(siamese_model).to(device)

    # define optimizer and criterion
    # only train the classifier
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, classifier_model.parameters()), lr=learning_rate)

    # define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Use `BCEWithLogitsLoss` as the loss function

    # get classifier dataset
    classifier_dataset = get_classifier_train_dataset('E:/comp3710/AD_NC')
    classifier_dataloader = DataLoader(classifier_dataset, batch_size=batch_size, shuffle=True)

    save_directory = "E:/PatternAnalysis-2023/results"
    save_filename = f"classifier_network_{epochs}epochs2.pt"

    # train the classifier
    for epoch in range(epochs):
        classifier_model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(classifier_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = classifier_model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))  # Make sure labels are the correct shape

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = torch.sigmoid(outputs).round()  # Get predictions from the outputs
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()

            if batch_idx % 10 == 0:  # Log after every 10 batches
                # Calculate accuracy
                accuracy = 100 * correct / total
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}%'.format(
                    epoch + 1, batch_idx * len(inputs), len(classifier_dataloader.dataset),
                    100. * batch_idx / len(classifier_dataloader), loss.item(), accuracy))

                # Log loss and accuracy to TensorBoard
                writer.add_scalar('Loss/train', running_loss / (batch_idx + 1),
                                  epoch * len(classifier_dataloader) + batch_idx)
                writer.add_scalar('Accuracy/train', accuracy, epoch * len(classifier_dataloader) + batch_idx)

        # Log epoch metrics
        writer.add_scalar('Loss/train_epoch', running_loss / len(classifier_dataloader), epoch)
        writer.add_scalar('Accuracy/train_epoch', 100 * correct / total, epoch)

    # create directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # save model
    save_path = os.path.join(save_directory, save_filename)
    torch.save(classifier_model.state_dict(), save_path)

    writer.close()  # Close the TensorBoard writer

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train_siamese(device)
    train_classifier(device, learning_rate=0.01)


