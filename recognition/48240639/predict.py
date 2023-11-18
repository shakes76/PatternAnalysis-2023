"""
Created on Wednesday October 18 

Siamese Network Testing Script

This script is used to test a Siamese Network model on a dataset. It loads a trained Siamese Network model, 
evaluates its performance on a test dataset, and reports the accuracy and loss.

@author: Aniket Gupta 
@ID: s4824063

"""

# Import necessary libraries
from dataset import get_testing  # Import the function for loading the test dataset
from modules import SiameseNN   # Import the Siamese Network model
import torch                     # Import the PyTorch library
from torch.utils.data import DataLoader # Import DataLoader for loading test data
from torch import nn             # Import the neural network module from PyTorch

# Define the testing function
def test(model, device, test_loader):
    """
    Test a Siamese Network model on a dataset.

    Parameters:
    - model: SiameseNN
        The Siamese Network model to be tested.
    - device: torch.device
        The device to run the testing on (e.g., CPU or GPU).
    - test_loader: DataLoader
        The DataLoader for the test dataset.

    Returns:
    None
    """
    print("Testing Started.")

    model.eval()  # Set the model to evaluation mode (no gradient computation)
    test_loss, correct, criterion = 0, 0, nn.BCELoss()  # Initialize variables

    with torch.no_grad():
        for (images_1, images_2, targets) in test_loader:
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()  # Forward pass to obtain model's predictions
            test_loss += criterion(outputs, targets).sum().item()  # Calculate the test loss
            pred = torch.where(outputs > 0.5, 1, 0)  # Convert model outputs to binary predictions
            correct += pred.eq(targets.view_as(pred)).sum().item()  # Count correct predictions

    test_loss /= len(test_loader.dataset)  # Calculate the average test loss
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset))
    )
    print("Finished testing.")

# Entry point of the script
if __name__ == '__main__':
    device, batch_size = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 256  # Define the device and batch size
    model = SiameseNN()  # Initialize the Siamese Network model
    model.load_state_dict(
        torch.load("/Users/aniketgupta/Desktop/Pattern Recognition/PatternAnalysis-2023/results/siam_net_50epochs.pt", map_location=torch.device('cpu'))
    )  # Load the pre-trained model's weights
    train_data = get_testing('/Users/aniketgupta/Desktop/Pattern Recognition/PatternAnalysis-2023/recognition/48240639/AD_NC')
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)  # Load the test dataset
    test(model, device, train_dataloader)  # Call the testing function to evaluate the model
