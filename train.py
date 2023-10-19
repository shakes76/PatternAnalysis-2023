import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from dataset import ISICDataset
from modules import UNet
from torch.utils.tensorboard import SummaryWriter
import gc
from utils import dice_coefficient
import matplotlib.pyplot as plt

# Function to save plots
def save_plot(data, filename, y_label):
    plt.figure(figsize=(10,6))
    plt.plot(data)
    plt.title(y_label)
    plt.ylabel(y_label)
    plt.xlabel('Epoch')
    plt.savefig(filename)
    plt.close()

# Hyperparameters
batch_size = 16
epochs = 10
learning_rate = 0.001
validation_split = 0.1  # use 10% of the dataset for validation

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])


# Set your paths
train_image_dir = 'C:\\Users\\25060\\Desktop\\ISIC-2017_Training_Data'

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Use the path for training dataset
dataset = ISICDataset(image_dir=train_image_dir, transform=transform)

# Split dataset into training and validation sets
dataset = ISICDataset(image_dir='data', transform=transform)
train_len = int((1-validation_split) * len(dataset))
valid_len = len(dataset) - train_len
train_dataset, valid_dataset = random_split(dataset, [train_len, valid_len])

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

model = UNet(in_channels=3, out_channels=1).to(device)
criterion = torch.nn.BCEWithLogitsLoss()  # Binary Cross Entropy for binary segmentation
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

# TensorBoard writer
writer = SummaryWriter()

best_val_loss = float('inf')

# Lists to store losses and dice scores
train_losses = []
valid_losses = []
valid_dices = []

# Training loop
for epoch in range(epochs):
    epoch_loss = 0
    model.train()
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
    model.eval()
    val_loss = 0
    val_dice = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()

            # Calculate Dice score for each batch
            output = torch.sigmoid(output) 
            predicted_mask = (output > 0.5).float()
            dice_score = dice_coefficient(predicted_mask, target)
            val_dice += dice_score

 # Calculate average loss and dice scores
    train_losses.append(epoch_loss / len(train_loader))
    valid_losses.append(val_loss / len(valid_loader))
    valid_dices.append(val_dice / len(valid_loader))

    # Update learning rate
    scheduler.step(val_loss/len(valid_loader))

    # Log losses and dice score to TensorBoard
    writer.add_scalar('Loss/Training', epoch_loss/len(train_loader), epoch)
    writer.add_scalar('Loss/Validation', val_loss/len(valid_loader), epoch)
    writer.add_scalar('Dice Score/Validation', val_dice/len(valid_loader), epoch)

    # Print epoch statistics
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss/len(train_loader)}, Validation Loss: {val_loss/len(valid_loader)}, Validation Dice: {val_dice/len(valid_loader)}") 

    # Save the model with the best validation loss
    if (val_loss/len(valid_loader)) < best_val_loss:
        best_val_loss = val_loss/len(valid_loader)
        torch.save(model.state_dict(), 'best_model.pth')

    # Save plots
    save_plot(train_losses, 'Traning_Loss.png', 'Training Loss')
    save_plot(valid_losses, 'Validation_Loss.png', 'Validation Loss')
    save_plot(valid_dices, 'Validation_Dice.png', 'Validation Dice Score')

    # Optional: Garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

writer.close()  # Close the TensorBoard writer
