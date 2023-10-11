import torch
import torch.nn as nn
import torch.optim as optim
from modules import ESPCN

# Create the model and load the data
model = ESPCN(upscale_factor=4, channels=1)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20