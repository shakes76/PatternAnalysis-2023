import torch
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import ISICDataset  # Assuming this is your dataset class
from modules import UNet  # Assuming this is your model class
from utils import dice_coefficient  # If you use this in your training loop

# Parsing command-line arguments
parser = argparse.ArgumentParser(description='Training script for the U-Net model.')
parser.add_argument('--train-dir', type=str, required=True, help='Path to the training data directory.')
parser.add_argument('--valid-dir', type=str, required=True, help='Path to the validation data directory.')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training.')
parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate.')
args = parser.parse_args()

# Dataset and DataLoader setup
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

train_dataset = ISICDataset(args.train_dir, transform=transform)
valid_dataset = ISICDataset(args.valid_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model, Loss, Optimizer setup
model = UNet(in_channels=3, out_channels=1).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# Training Loop
train_losses = []
valid_losses = []

for epoch in range(args.epochs):
    model.train()
    epoch_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    train_losses.append(epoch_loss)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()

    valid_losses.append(val_loss)

    # Optional: Print epoch statistics, calculate Dice coefficient, etc.

    # Clear unused memory
    torch.cuda.empty_cache()

# Save the model
torch.save(model.state_dict(), 'model.pth')
