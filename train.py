import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from dataset import ISICDataset
from modules import UNet
from torch.utils.tensorboard import SummaryWriter
import gc
from utils import dice_coefficient


# Hyperparameters
batch_size = 16
epochs = 10
learning_rate = 0.001
validation_split = 0.1  # use 10% of the dataset for validation

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

# Split dataset into training and validation sets
dataset = ISICDataset(root_dir='data', transform=transform)
train_len = int((1-validation_split) * len(dataset))
valid_len = len(dataset) - train_len
train_dataset, valid_dataset = random_split(dataset, [train_len, valid_len])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

model = UNet(in_channels=3, out_channels=1).to(device)
criterion = torch.nn.BCEWithLogitsLoss()  # Binary Cross Entropy for binary segmentation
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

# TensorBoard writer
writer = SummaryWriter()

best_val_loss = float('inf')

# Training loop
model.train()

for epoch in range(epochs):
    epoch_loss = 0
    model.train()  # Ensure the model is in training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # Compute loss
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    val_dice = 0  
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()

    # Calculate and accumulate the Dice score
    output = torch.sigmoid(output) 
    predicted_mask = (output > 0.5).float()
    dice_score = dice_coefficient(predicted_mask, target)
    val_dice += dice_score
    
    # Compute average Dice score for the entire validation set
    val_dice /= len(valid_loader) 

    
    # Update learning rate
    scheduler.step(val_loss_avg)

    # Log losses to TensorBoard
    writer.add_scalar('Loss/Training', train_loss_avg, epoch)
    writer.add_scalar('Loss/Validation', val_loss_avg, epoch)

    # Print epoch statistics
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss/len(train_loader)}, Validation Loss: {val_loss/len(valid_loader)}, Validation Dice: {val_dice}") 
    # Save the model with the best validation loss
    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        torch.save(model.state_dict(), 'best_model.pth')

    # Optional: Garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

writer.close()  # Close the TensorBoard writer
