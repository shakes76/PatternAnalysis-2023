import os
import torch
from train import (get_data_loaders, initialize_model, 
                                train_one_epoch, validate_one_epoch,
                                save_sample_images, plot_training_progress)

# Directories 
output_dir = "C:/users/lombo/Desktop" # Output directory
image_dir = "C:/Users/lombo/Desktop/3710_report/ISIC2018_Task1-2_Test_Input/ISIC2018_Task1-2_Test_Input" # Image directory
mask_dir = "C:/Users/lombo/Desktop/3710_report/ISIC2018_Task1_Test_GroundTruth" # Mask directory

# Parameters
n_batch = 128
learning_rate = 0.001
n_epochs = 30

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Loaders
    train_loader, val_loader = get_data_loaders(image_dir, mask_dir, n_batch)
    
    # Initialize Model, Criterion, and Optimizer
    model, criterion, optimizer = initialize_model(device, learning_rate)
    
    # Lists to keep track of progress
    train_loss_history, train_dice_history = [], []
    val_loss_history, val_dice_history = [], []

    # Main Training Loop
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        print('-' * 20)
        
        # Train for one epoch
        train_loss, train_dice = train_one_epoch(model, criterion, optimizer, train_loader, device)
        train_loss_history.append(train_loss)
        train_dice_history.append(train_dice)
        
        # Validate for one epoch
        val_loss, val_dice = validate_one_epoch(model, criterion, val_loader, device)
        val_loss_history.append(val_loss)
        val_dice_history.append(val_dice)
        
        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        # Save some sample images for visualization
        save_sample_images(epoch, val_loader, model, output_dir, device)

    # Plot training progress
    plot_training_progress(train_loss_history, val_loss_history, train_dice_history, val_dice_history, output_dir)

if __name__ == "__main__":
    main()
