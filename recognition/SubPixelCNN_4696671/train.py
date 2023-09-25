"""
TRAINING LOOP
"""

# Imports
from dataset import get_test_loader, get_train_loader
from modules import ESPCN
import torch
import torchvision.utils
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import math

# PyTorch setup
print("PyTorch Version:", torch.__version__)
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.stdout.flush()

# Get data loaders
train_loader = get_train_loader()
test_loader = get_test_loader()

# downscale
def downscale(images, factor=4):
    return torch.tensor(map(lambda x: Resize(tuple(map(lambda y: y // factor, original_size)))(x),images))

# Vizualise some of the training data
real_batch = next(iter(train_loader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images - Targets")
plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

# Get Model
model = ESPCN(1, 4)
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

original_size = (240, 256)


# Training Loop
model.train()
print("> Training")
sys.stdout.flush()
start = time.time()
for epoch in range(epochs):
    for i, (data, _) in enumerate(train_loader):
        
        # Send data to device
        data = data.to(device)

        # Downscale images by factor of 4
        new_data = downscale(data)

        # Forward pass of model
        outputs = model(new_data)
        loss = criterion(outputs, data)

        # Optimization step
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if (i+1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:.5f}"
                  .format(epoch+1, epochs, i+1, total_step))
            sys.stdout.flush()

        # Decay Learning Rate using Scheduler            
        scheduler.step()

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")


