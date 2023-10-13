import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset import SuperResolutionDataset
from modules import ESPCN
from torch.cuda.amp import GradScaler, autocast
# will add in future

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 100
LR = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create dataset and data loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = SuperResolutionDataset(root_dir='AD_NC', transform=transform, mode='train')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model, Loss function, Optimizer
model = ESPCN().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training and validation functions
def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for lr, hr in loader:
        lr, hr = lr.to(DEVICE), hr.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(lr)
        loss = criterion(outputs, hr)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            outputs = model(lr)
            loss = criterion(outputs, hr)
            running_loss += loss.item()
    return running_loss / len(loader)

# Main training loop
train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, criterion, optimizer)
    val_loss = validate(model, val_loader, criterion)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')

# Plotting the training and validation losses
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
