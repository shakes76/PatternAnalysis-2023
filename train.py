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
def show_plot(data, y_label):
    plt.figure(figsize=(10,6))
    plt.plot(data)
    plt.title(y_label)
    plt.ylabel(y_label)
    plt.xlabel('Epoch')
    plt.show()
    plt.close()

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset setup
train_image_dir = 'C:\\Users\\25060\\Desktop\\ISIC-2017_Training_Data'
valid_image_dir = 'C:\\Users\\25060\\Desktop\\ISIC-2017_Validation_Data'
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

train_dataset = ISICDataset(train_image_dir, transform=transform)
valid_dataset = ISICDataset(valid_image_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

# Model, Loss, Optimizer setup
model = UNet(in_channels=3, out_channels=1).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 10
train_losses = []
valid_losses = []

for epoch in range(epochs):
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

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()

    # Store average losses for plotting
    train_losses.append(epoch_loss / len(train_loader))
    valid_losses.append(val_loss / len(valid_loader))

    # Print epoch statistics
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss/len(train_loader)}, Validation Loss: {val_loss/len(valid_loader)}")

# Plotting losses
show_plot(train_losses, 'Training Loss')
show_plot(valid_losses, 'Validation Loss')

    # Optional: Garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

writer.close()  # Close the TensorBoard writer
