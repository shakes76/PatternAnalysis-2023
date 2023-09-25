"""
LOADING AND TRANSFORMING THE DATA
"""

# Imports
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

# Data Loading Parameters
dataroot = ".\\ADNI\\AD_NC"
batch_size = 128

# Setup Data Transforms
transform = transforms.Compose([
    transforms.ToTensor()
])

# Create data set from the data in the root
dataset = datasets.ImageFolder(root=dataroot,transform=transform)

# Create dataloader from dataset
dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Function to get dataloader from other files
def getDataLoader():
    return dataLoader





