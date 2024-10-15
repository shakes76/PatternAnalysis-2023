import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

from modules import ImprovedUNET
from dataset import train_images, val_images, test_images

# Dice loss function
class DiceLoss(nn.Module):
    """
    Custom PyTorch module for computing the Dice loss.

    It measures the dissimilarity between the predicted output and the target,
    based on the overlap between the predicted and target binary masks.
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, output, target):
        """
        Compute the Dice loss between the predicted output and the target.
        """
        smooth = 1.0
        intersection = torch.sum(output * target)
        union = torch.sum(output) + torch.sum(target)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice
    
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")


def train_unet_with_dice_loss(data, labels, num_epochs=300, batch_size=2, lr_init=1e-4, weight_decay=1e-5):
    """
    Train a U-Net model using the Dice loss.
    """
    # Create a DataLoader for the dataset
    dataloader = DataLoader(val_images, batch_size=batch_size, shuffle=True)

    # Define the loss function (Dice loss) and optimizer
    criterion = DiceLoss()
    optimizer = optim.Adam(ImprovedUNET.parameters(), lr=lr_init, weight_decay=weight_decay)

    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, (input_data, target) in enumerate(dataloader):
            ImprovedUNET.train()
            optimizer.zero_grad()
            output = ImprovedUNET(input_data)

            # Compute the Dice loss
            loss = criterion(output, target)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item()}")

