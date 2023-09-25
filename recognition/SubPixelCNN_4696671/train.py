"""
TRAINING LOOP
"""

# Imports
from dataset import getDataLoader
import torch
import torchvision.utils
import matplotlib.pyplot as plt
import numpy as np
import sys

# PyTorch setup
print("PyTorch Version:", torch.__version__)
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.stdout.flush()

# Vizualise some of the training data
dataLoader = getDataLoader()

real_batch = next(iter(dataLoader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()