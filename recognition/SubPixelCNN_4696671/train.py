"""
TRAINING LOOP
"""

# Imports
from dataset import get_test_loader, get_train_loader
from modules import ESPCN
import torch
import torchvision.utils
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

# PyTorch setup
print("PyTorch Version:", torch.__version__)
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.stdout.flush()

# Get data loaders
train_loader = get_train_loader()
test_loader = get_test_loader()

# Get Model
model = ESPCN(1, 4)
model = model.to(device)

# Vizualise some of the training data
real_batch = next(iter(train_loader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

# Training Parameters
epochs = 10
learning_rate = 0.001
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, 
                                                max_lr=learning_rate, 
                                                epochs=epochs,
                                                steps_per_epoch=391)

# Training Loop
model.train()
print("> Training")
sys.stdout.flush()
start = time.time()
for epoch in range(epochs):
    for i, data in enumerate(train_loader, 0):
        pass