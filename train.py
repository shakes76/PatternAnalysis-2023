import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms  
from torch.utils.data import DataLoader
from modules import UNetPlusPlus, DiceLoss  # Import UNetPlusPlus model
from dataset import CustomDataset

def train(model, train_loader, criterion, optimizer):
    """
    Training loop function. It trains the model using the provided data loader.

    Args:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for the training dataset.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in train_loader:
        inputs, masks = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    return avg_loss

def validate(model, val_loader, criterion):
    """
    Validation loop function. It evaluates the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (nn.Module): Loss function.

    Returns:
        float: Average validation loss for the epoch.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs, masks = batch
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    return avg_loss

# Set hyperparameters and paths
data_dir = "path/to/your/dataset"
batch_size = 16
learning_rate = 0.001
num_epochs = 10

# Instantiate the UNet++ model and the dataset
in_channels = 3  # Adjust based on your dataset
out_channels = 1  # Adjust based on your segmentation task
model = UNetPlusPlus(in_channels, out_channels, num_levels=4)  # Create your UNet++ model
transform = transforms.Compose([transforms.ToTensor()])  # Adjust as needed
dataset = CustomDataset(data_dir, transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define loss function and optimizer
criterion = DiceLoss()  # Use your appropriate loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == "__main__":
    for epoch in range(num_epochs):
        model.train()
        train_loss = train(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}")

        model.eval()
        val_loss = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "trained_model.pth")
