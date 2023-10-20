import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

from modules import ImprovedUNET
from dataset import train_images, val_loader, test_images

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 500
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False

# Define the Dice loss function
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, output, target):
        smooth = 1.0
        intersection = torch.sum(output * target)
        union = torch.sum(output) + torch.sum(target)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice
    
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")



def train_unet_with_dice_loss(data, labels, num_epochs=300, batch_size=2, lr_init=5e-4, weight_decay=1e-5):
    # Create a DataLoader for the dataset
    dataloader = DataLoader(val_loader, batch_size=batch_size, shuffle=True)

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

if __name__ == "__main__":
    # Load your data and labels here
    data = np.random.randn(300, 1, 128, 128, 128)  # Replace with your actual data
    labels = np.random.randint(0, 2, (300, 1, 128, 128, 128))  # Replace with your actual labels

    train_unet_with_dice_loss(data, labels)

