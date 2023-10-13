
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from modules import ESPCN
from dataset import ADNIDataset, get_dataloaders, image_transform
import math

# Hyperparameters
learning_rate = 0.01
num_epochs = 5
upscale_factor = 4
batch_size = 32

# Initialize the model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ESPCN(upscale_factor=upscale_factor).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Load the data
train_loader, test_loader = get_dataloaders("C:\\Users\\soonw\\ADNI\\AD_NC", batch_size=batch_size)

# Training loop
losses = []
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_idx, (downsampled, original, _) in enumerate(train_loader):
        downsampled, original = downsampled.to(device), original.to(device)

        # Forward pass
        outputs = model(downsampled)
        loss = criterion(outputs, original)
        epoch_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    losses.append(epoch_loss / len(train_loader))


        # Evaluation on test data
    model.eval()
    with torch.no_grad():
        avg_psnr = 0
        for batch_idx, (downsampled, original, _) in enumerate(test_loader):
            downsampled, original = downsampled.to(device), original.to(device)
            outputs = model(downsampled)
            mse = criterion(outputs, original)
            psnr = 10 * math.log10(1 / mse.item())
            avg_psnr += psnr
        avg_psnr = avg_psnr / len(test_loader)
        print(f"Average PSNR on test data: {avg_psnr:.4f} dB")

# Save the trained model
#torch.save(model.state_dict(), 'espcn_model.pth')

# Plot training losses
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()


