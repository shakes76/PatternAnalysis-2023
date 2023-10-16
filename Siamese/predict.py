"""
    File name: modules.py
    Author: Fanhao Zeng
    Date created: 11/10/2023
    Date last modified: 16/10/2023
    Python Version: 3.10.12
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import get_test_dataset
from modules import SiameseNetwork


def test(model, device, test_loader):

    print("Starting testing.")
    model.eval()
    test_loss = 0
    correct = 0

    # Using `BCELoss` as the loss function
    criterion = nn.BCELoss()

    # No need to calculate gradients
    with torch.no_grad():
        for (images_1, images_2, targets) in test_loader:
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()
            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
            pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item() # Update correct count

    test_loss /= len(test_loader.dataset) # Calculate average loss

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    print("Finished testing.")


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 256

    model = SiameseNetwork()
    model_path = "../results/siamese_network_50epochs.pt"
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    print("Loading data...")
    train_data = get_test_dataset('E:/comp3710/AD_NC')
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    print("Data loaded.")

    test(model, device, train_dataloader)

