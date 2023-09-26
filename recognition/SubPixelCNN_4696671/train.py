"""
TRAINING LOOP
"""

# Imports
from dataset import get_test_loader, get_train_loader, downscale
from modules import ESPCN
import torch
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import math

# Constants
CHANNELS = 3
FACTOR = 4

# PyTorch setup
print("PyTorch Version:", torch.__version__)
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.stdout.flush()

# Get data loaders
train_loader = get_train_loader()
test_loader = get_test_loader()

# Vizualise some of the training data
image = next(iter(train_loader))[0][0]
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Sample Training Image - Original")
plt.imshow(image.permute(1,2,0), cmap='gray')

plt.subplot(1,2,2)
plt.axis("off")
plt.title("Sample Training Image - Downsampled")
plt.imshow(downscale(image, FACTOR).permute(1,2,0), cmap='gray')

plt.savefig("sample_input.png")


# Get Model
model = ESPCN(CHANNELS, FACTOR)
model = model.to(device)

# Training Parameters
epochs = 10
learning_rate = 0.001
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, 
                                                max_lr=learning_rate, 
                                                epochs=epochs,
                                                steps_per_epoch=169)
criterion = torch.nn.functional.mse_loss
total_step = len(train_loader)

losses = []

# Training Loop
model.train()
print("> Training")
sys.stdout.flush()

start = time.time()
for epoch in range(epochs):
    for i, (data, _) in enumerate(train_loader):
        
        # Generate model inputs via downsampling
        new_data = downscale(data, FACTOR)

        # Send data to device
        data = data.to(device)
        new_data = new_data.to(device)

        # Forward pass of model
        outputs = model(new_data)
        loss = criterion(outputs, data)
        losses.append(loss.item())

        # Optimization step
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if (i+1) % 30 == 0:
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:.5f}"
                  .format(epoch+1, epochs, i+1, total_step, loss.item()))
            sys.stdout.flush()

        # Decay Learning Rate using Scheduler            
        scheduler.step()

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")


# Testing
print("> Testing")
sys.stdout.flush()


start = time.time()
model.eval()
psnrs = []
test_losses = []

def PSNR(mse, maxi = 1):
    # Calculate PSNR (default maxi is 1 because float representation of color)
    return 10 * math.log(maxi**2 / mse, 10)

with torch.no_grad():
    
    for i, (data, _) in enumerate(test_loader):

        data = data.to(device)

        # Downscale data by factor
        new_data = downscale(data, FACTOR)

        # Forward model pass
        outputs = model(new_data)
        mse = criterion(outputs, data).item()
        
        test_losses.append(mse)
        psnrs.append(PSNR(mse, 1))

print("Average PSNR on Test Set: " + str(np.mean(psnrs)))
print("Average Loss on Test Set: " + str(np.mean(test_losses)))
sys.stdout.flush()

# Save trained Model
torch.save(model.state_dict(), "model.pth")

# Plot loss per step
plt.figure(figsize=(10,5))
plt.title("MSE Loss during training")
plt.plot(losses)
plt.xlabel("step")
plt.ylabel("loss")
plt.savefig("losses.png")