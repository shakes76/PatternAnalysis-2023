import torch.optim as optim
from dataset import Dataset
from modules import Unet

# Load model and dataset
modules = Unet()
dataset = Dataset()

# Load optimizer
optimizer = optim.Adam(modules.parameters(), lr=0.001)