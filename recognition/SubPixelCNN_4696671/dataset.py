"""
LOADING AND TRANSFORMING THE DATA
"""

# Imports
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

# Data Loading Parameters
dataroot = ".\\ADNI\\AD_NC"
test_suffix = "\\test"
train_suffix = "\\train"
batch_size = 128

# Setup Data Transforms
transform = transforms.Compose([
    transforms.ToTensor()
])

# Create datasets from the data in the root
train_dataset = datasets.ImageFolder(root=dataroot+train_suffix, transform=transform)
test_dataset = datasets.ImageFolder(root=dataroot+test_suffix, transform=transform)

# Create dataloaders from dataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Function to get dataloader from other files
def get_train_loader():
    return train_loader

def get_test_loader():
    return test_loader





