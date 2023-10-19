"""
Created on Wednesday October 18 

Siamese Network Testing Script

This script is used to test a Siamese Network model on a dataset. It loads a trained Siamese Network model, 
evaluates its performance on a test dataset, and reports the accuracy and loss.

@author: Aniket Gupta 
@ID: s4824063

"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import get_testing
from modules import SiameseNN


def test(model, device, test_loader):

    print("Testing Started.")

    model.eval()
    test_loss, correct, criterion = 0, 0, nn.BCELoss()

    with torch.no_grad():
        for (images_1, images_2, targets) in test_loader:
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()
            test_loss += criterion(outputs, targets).sum().item()
            pred = torch.where(outputs > 0.5, 1, 0)
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    print("Finished testing.")


if __name__ == '__main__':
    device, batch_size = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 256
    model = SiameseNN()
    model.load_state_dict(torch.load("/Users/aniketgupta/Desktop/Pattern Recognition/PatternAnalysis-2023/results/siamese_network_50epochs.pt", map_location=torch.device('cpu')))
    train_data = get_testing('/Users/aniketgupta/Desktop/Pattern Recognition/PatternAnalysis-2023/recognition/48240639/AD_NC')
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test(model, device, train_dataloader)