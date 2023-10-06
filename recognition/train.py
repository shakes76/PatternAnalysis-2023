import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import ISICDataset, get_transform
from modules import UNet3D
from modules2 import build_unet

# Hyperparameters
lr = 0.001  # Learning rate
batch_size = 2
num_epochs = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialise dataset and dataloaders
train_dataset = ISICDataset(dataset_type='training', transform=get_transform())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model, Loss, and Optimiser
model = UNet3D(in_channels=3, num_classes=2).to(device)
# model = build_unet().to(device)
optimiser = optim.Adam(model.parameters(), lr=lr)

# Training Loop
model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    
    for i, sample in enumerate(train_loader):
        inputs, labels = sample['image'].to(device), sample['mask'].to(device)
        
        # Zero the parameter gradients
        optimiser.zero_grad()
        
        # Forward pass
        outputs = model(inputs)

        print("here")
        
        # Loss computation using cross entropy loss
        # Ensure your mask labels are of type torch.long
        labels = labels.long()
        loss = F.cross_entropy(outputs, labels)
        
        # Backward pass and optimisation
        loss.backward()
        optimiser.step()
        
        total_loss += loss.item()
        
        # Print statistics
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {total_loss / (i + 1):.4f}")
