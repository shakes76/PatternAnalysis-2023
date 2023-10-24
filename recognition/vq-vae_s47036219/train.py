# CONSTANTS AND HYPERPARAMETERS:

import torch
import torch.nn as nn
import torch.optim as optim
from modules import VQVAE, device, ssim
from dataset import get_dataloaders



LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 40 # realistically stopped earlier by the validation set
CODEBOOK_SIZE = 512

# Weights for the loss functions
L2_WEIGHT = 0.05
SSIM_WEIGHT = 1

# Constants for early stopping
PATIENCE = 12
best_val_loss = float('inf')
counter = 0


def train(vqvae, train_loader, validation_loader):
    optimizer = optim.Adam(vqvae.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
    # Training Loop
    for epoch in range(NUM_EPOCHS):
        for i, (img, _) in enumerate(train_loader):
            vqvae.train()
            img = img.to(device)  # Move to device

            # Forward pass through the entire model
            recon = vqvae(img)
            
            # Calculate L2 loss
            l2_loss = ((recon - img) ** 2).sum()

            # Calculate SSIM loss
            ssim_loss = 1 - ssim(img, recon)

            # Final Loss
            loss = L2_WEIGHT * l2_loss + SSIM_WEIGHT * ssim_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation phase
        val_losses = []
        vqvae.eval()
        with torch.no_grad():
            for i, (img, _) in enumerate(validation_loader):
                img = img.to(device)
                
                # Validation forward pass
                recon = vqvae(img)  # Changed this line to use the VQVAE model
                
                # Validation losses
                l2_loss = ((recon - img) ** 2).sum()
                ssim_loss = 1 - ssim(img, recon)
                loss = L2_WEIGHT * l2_loss + SSIM_WEIGHT * ssim_loss

                val_losses.append(loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Training Loss: {loss.item():.4f}, Validation Loss: {avg_val_loss:.4f}")

        # Update learning rate
        scheduler.step(avg_val_loss)
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0  # Reset counter when validation loss decreases
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    torch.save(vqvae.state_dict(), 'vqvae.pth')
    
def train_new_model(train, validation): # Called if weight didnt exist in the test set.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on: ", device)
    model = VQVAE(CODEBOOK_SIZE).to(device)
    model = train(model, train, validation)
    
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on: ", device)
    train_loader, validation_loader, _ = get_dataloaders(BATCH_SIZE)
    
    model = VQVAE(CODEBOOK_SIZE).to(device)
    model = train(model, train_loader, validation_loader)
    
if __name__ == "__main__":
    main()