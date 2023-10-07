import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ISICDataset, get_transform
from modules2 import UNet2D

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
train_dataset = ISICDataset(dataset_type='training', transform=get_transform())
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
for i, data in enumerate(train_loader, 0):
    inputs, labels = data['image'], data['mask']
    print(f"Input shape: {inputs.shape}, Labels shape: {labels.shape}")
    break

# Initialize model, optimizer, and loss function
model = UNet2D(3, 1).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Training Function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data['image'].to(device), data['mask'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Log Information
        if i % 10 == 9:  # print every 10 mini-batches
            print(f"[Step {i+1}] Loss: {loss.item():.4f}")


    return running_loss / len(dataloader)

# Training Loop
num_epochs = 25
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")

# Save Model Weights
torch.save(model.state_dict(), 'unet_model.pth')
