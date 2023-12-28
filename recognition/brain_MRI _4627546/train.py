"""
train.py

Student Name: Zijun Zhu
Student ID: s4627546
Bref intro:
containing the source code for training, validating, testing and saving your model. The model
should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
sure to plot the losses and metrics during training
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast  # Mixed precision training
import matplotlib.pyplot as plt
from dataset import SuperResolutionDataset
from modules import ESPCN

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 200
LR = 0.0005
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# new! data enhancement
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataset = SuperResolutionDataset(root_dir='AD_NC', transform=train_transform, mode='train')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model, Loss function, Optimizer
model = ESPCN().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Dynamic learning rate scheduling
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)

# Mixed precision training
scaler = GradScaler()


# Update: Training and validation functions with mixed precision
def train(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    for lr, hr in loader:
        lr, hr = lr.to(DEVICE), hr.to(DEVICE)
        optimizer.zero_grad()

        with autocast():
            outputs = model(lr)
            loss = criterion(outputs, hr)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
print('training start')
for epoch in range(EPOCHS):
    start_time = time.time()

    train_loss = train(model, train_loader, criterion, optimizer, scaler)
    val_loss = validate(model, val_loader, criterion)

    end_time = time.time()
    epoch_time = end_time - start_time  # Calculate the time taken for the epoch

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch + 1}/{EPOCHS} - Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f},"
          f" Time: {epoch_time:.2f} seconds")

    scheduler.step()

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')

# Plotting the training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='red')
plt.ylim([0, max(train_losses + val_losses) + 0.05])  # Adjusting the y-axis range
plt.legend()
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.show()
