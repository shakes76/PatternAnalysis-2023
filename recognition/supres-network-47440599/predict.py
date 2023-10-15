import torch
import torchvision
import torchvision.utils as vutils
from modules import SubPixel
from dataset import *


device = torch.device("cpu")

# Define model parameters


# Instantiate the model
model = SubPixel()  # Use the correct parameters

# Load the pre-trained model's state dict
model.load_state_dict(torch.load("subpixel_model.pth", map_location=device))
print(model)


# Generate random noise for input

# Generate fake images with the model


# Visualize the generated fake images

with torch.no