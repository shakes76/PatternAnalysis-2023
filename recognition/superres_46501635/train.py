import torch
import torch.optim as optim
from modules import ESPCN
from dataset import get_dataloaders
import math
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate_initial = 0.01
learning_rate_final = 0.0001
epochs = 20  # You can adjust this based on your needs
upscale_factor = 4  # Adjust based on your needs

# Load datasets
train_loader, test_loader = get_dataloaders("C:\\Users\\soonw\\ADNI\\AD_NC")  # Replace path based on your folder location

# Initialize model and move to device
model = ESPCN(upscale_factor).to(device)

# Loss and optimizer
criterion = torch.nn.MSELoss() # MSE Loss used in many image reconstruction tasks. Super-res = make reconstructed image as similar to original
                               # as possible. It's used to ensure that reconstructed high-res image is pixel-wise similar to the original
optimizer = optim.Adam(model.parameters(), lr=learning_rate_initial)
# Literature states that for deep learning tasks including super-resolution, Adam optimiser has been found to converge faster and achieve
# better performance than traditional SGD. Its adaptive learning rate and momentum terms help in navigating the loss landscape more effectively.

# Learning rate scheduler to adjust learning rate during training
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=learning_rate_final)
# based on literature

# Function to compute PSNR
def compute_psnr(mse_loss):
    # Peak Signal-to-Noise Ratio (PSNR) is a widely used metric in image processing, 
    # especially for tasks like image compression and super-resolution. It measures the quality of the reconstructed image 
    # compared to the original image. A higher PSNR indicates better quality. It's particularly useful for super-resolution 
    # because it gives an indication of how well the model has enhanced the image while preserving the original details.
    return 10 * math.log10(1 / mse_loss)

# Lists to store training and evaluation data
train_losses = []
test_avg_psnrs = []

# Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch_idx, (LR, HR, _) in enumerate(train_loader):
        LR, HR = LR.to(device), HR.to(device)

        # Forward pass
        outputs = model(LR)
        loss = criterion(outputs, HR)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Calculate average training loss for the epoch
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Adjust learning rate
    scheduler.step(avg_train_loss)

    # Evaluation on the test set
    model.eval()
    avg_psnr = 0
    with torch.no_grad():
        for batch_idx, (LR, HR, _) in enumerate(test_loader):
            LR, HR = LR.to(device), HR.to(device)
            outputs = model(LR)
            mse_loss = criterion(outputs, HR).item()
            avg_psnr += compute_psnr(mse_loss)

        avg_psnr /= len(test_loader)
        test_avg_psnrs.append(avg_psnr)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_train_loss:.4f}, Avg PSNR: {avg_psnr:.4f}")

# Plot the loss curve
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), train_losses, label='Training Loss', marker='o')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot the PSNR curve
plt.subplot(1, 2, 2)
plt.plot(range(epochs), test_avg_psnrs, label='Avg PSNR (Test Set)', marker='o')
plt.title('Average PSNR on Test Set')
plt.xlabel('Epoch')
plt.ylabel('Avg PSNR')
plt.legend()

# Save the plots
plt.savefig('training_plots.png')
plt.show()

print("Training finished.")
# Save the trained model
save_path = 'C:\\Users\\soonw\\ADNI\\ESPCN_trained_model.pth'
torch.save(model.state_dict(), save_path)