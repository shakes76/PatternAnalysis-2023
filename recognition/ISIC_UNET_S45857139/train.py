import os
import torch
import torch.optim as optim
from torch.utils.data import random_split
from modules import UNETImproved
from dataset import get_isic_dataloader
from torchvision.utils import save_image
import matplotlib.pyplot as plt

def get_data_loaders(image_dir, mask_dir, batch_size=128):
    """
    Splits the ISIC dataset into training and validation sets and creates corresponding DataLoaders.

    Args:
        - image_dir (str): Path to the directory with image files.
        - mask_dir (str): Path to the directory with mask files.
        - batch_size (int, optional): Number of samples per batch. Defaults to 128.

    Returns:
        - tuple: Training and validation DataLoaders.
    """
    full_loader = get_isic_dataloader(image_dir, mask_dir, batch_size=batch_size)
    train_size = int(0.8 * len(full_loader.dataset))
    val_size = len(full_loader.dataset) - train_size
    train_dataset, val_dataset = random_split(full_loader.dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def initialize_model(device, lr=0.001):
    """
    Initializes and returns the UNETImproved model, Binary Cross Entropy loss, and Adam optimizer.

    Args:
        - device (torch.device): Device (cpu or cuda) to which the model should be moved.
        - lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.

    Returns:
        - tuple: Model, criterion (loss function), and optimizer.
    """
    model = UNETImproved().to(device)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer

def dice_coefficient(prediction, target):
    """
    Computes the Dice coefficient between predicted and target masks.

    Args:
        - prediction (torch.Tensor): Predicted mask tensor.
        - target (torch.Tensor): Ground truth mask tensor.

    Returns:
        - float: Dice coefficient.
    """
    num = prediction.size(0)
    x = prediction.view(num, -1).float()
    y = target.view(num, -1).float()
    intersect = (x * y).sum().float()
    return (2 * intersect) / (x.sum() + y.sum())

def train_one_epoch(model, criterion, optimizer, train_loader, device):
    """
    Performs one epoch of training on the provided model using the training data.

    Args:
        - model (nn.Module): Model to be trained.
        - criterion (nn.Module): Loss function.
        - optimizer (torch.optim.Optimizer): Optimizer.
        - train_loader (DataLoader): DataLoader for training data.
        - device (torch.device): Device (cpu or cuda) on which computations should be performed.

    Returns:
        - tuple: Average training loss and average dice coefficient for the epoch.
    """
    print("Training")
    model.train()
    train_loss = 0.0
    train_dice_total = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        dice_val = dice_coefficient(outputs, labels)
        train_dice_total += dice_val.item()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader), train_dice_total / len(train_loader)

def validate_one_epoch(model, criterion, val_loader, device):
    """
    Performs one epoch of validation on the provided model using the validation data.

    Args:
        - model (nn.Module): Model to be validated.
        - criterion (nn.Module): Loss function.
        - val_loader (DataLoader): DataLoader for validation data.
        - device (torch.device): Device (cpu or cuda) on which computations should be performed.

    Returns:
        - tuple: Average validation loss and average dice coefficient for the epoch.
    """
    print("Validating...")
    model.eval()
    val_loss = 0.0
    val_dice_total = 0.0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            dice_val = dice_coefficient(outputs, labels)
            val_dice_total += dice_val.item()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(val_loader), val_dice_total / len(val_loader)

def save_sample_images(epoch, val_loader, model, output_dir, device, samples=3):
    """
    Saves sample images (input, label, prediction) for visualization.

    Args:
        - epoch (int): Current epoch number.
        - val_loader (DataLoader): DataLoader for validation data.
        - model (nn.Module): Model used for prediction.
        - output_dir (str): Directory where images should be saved.
        - device (torch.device): Device (cpu or cuda) on which computations should be performed.
        - samples (int, optional): Number of samples to save. Defaults to 3.
    """
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            if i == samples:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Saving input, mask, and output separately
            save_image(inputs[0].cpu(), os.path.join(output_dir, f"input_{epoch}_{i}.png"))
            save_image(labels[0].cpu(), os.path.join(output_dir, f"mask_{epoch}_{i}.png"))
            save_image(outputs[0].cpu(), os.path.join(output_dir, f"output_{epoch}_{i}.png"))

def plot_training_progress(train_loss_history, val_loss_history, train_dice_history, val_dice_history, output_dir):
    """
    Plots and saves training and validation progress in terms of loss and dice coefficient.

    Args:
        - train_loss_history (list): List of average training losses per epoch.
        - val_loss_history (list): List of average validation losses per epoch.
        - train_dice_history (list): List of average training dice coefficients per epoch.
        - val_dice_history (list): List of average validation dice coefficients per epoch.
        - output_dir (str): Directory where the plot should be saved.
    """
    plt.figure(figsize=(12, 6))

    # Plotting Training and Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Training Loss', color='blue')
    plt.plot(val_loss_history, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss per Epoch')

    # Plotting Training and Validation Dice Coefficient
    plt.subplot(1, 2, 2)
    plt.plot(train_dice_history, label='Training Dice Coefficient', color='blue')
    plt.plot(val_dice_history, label='Validation Dice Coefficient', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    plt.title('Dice Coefficient per Epoch')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_progress.png'))
    plt.show()

def main():
    """
    Main function to train and validate the Improved UNET model.

    This function sets up the training and validation environment, loads the ISIC dataset, initializes the model,
    and runs the training and validation loops. It also saves sample images and plots of the performance.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define directories and parameters
    image_dir = "/content/drive/My Drive/ISIC/IMAGE"  # Replace with your image directory
    mask_dir = "/content/drive/My Drive/ISIC/MASK"    # Replace with your mask directory
    save_dir = "/content/drive/My Drive/ISIC/Model" # Replace with your saved model directory
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 10  # Define the number of epochs
    output_dir = "/content/drive/My Drive"  # Directory to save output images and plots

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get data loaders
    train_loader, val_loader = get_data_loaders(image_dir, mask_dir, batch_size)

    # Initialize model, criterion, and optimizer
    model, criterion, optimizer = initialize_model(device, learning_rate)

    # Lists for storing loss and dice scores for each epoch
    train_loss_history = []
    val_loss_history = []
    train_dice_history = []
    val_dice_history = []

    for epoch in range(num_epochs):
        # Training
        train_loss, train_dice = train_one_epoch(model, criterion, optimizer, train_loader, device)
        train_loss_history.append(train_loss)
        train_dice_history.append(train_dice)

        # Validation
        val_loss, val_dice = validate_one_epoch(model, criterion, val_loader, device)
        val_loss_history.append(val_loss)
        val_dice_history.append(val_dice)

        # Save sample images
        save_sample_images(epoch, val_loader, model, output_dir, device)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

    # Save model
    model_save_path = os.path.join(save_dir, "model.pth")

    # Create the directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the model
    torch.save(model.state_dict(), model_save_path)

    # Plot training progress
    plot_training_progress(train_loss_history, val_loss_history, train_dice_history, val_dice_history, output_dir)

if __name__ == '__main__':
    main()
