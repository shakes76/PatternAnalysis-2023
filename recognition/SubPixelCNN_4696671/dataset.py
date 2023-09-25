"""
LOADING AND TRANSFORMING THE DATA
"""

# Imports
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.datasets as datasets

# Data Loading Parameters
dataroot = ".\\ADNI\\AD_NC"
test_suffix = "\\test"
train_suffix = "\\train"
ad_suffix = "\\AD"
nc_suffix = "\\NC"
batch_size = 128

# Create Data Transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# Function to get dataloader from other files
def get_train_loader():
    # Load data
    train_dataset_AD = datasets.ImageFolder(root=dataroot+train_suffix+ad_suffix, transform=transform)
    train_dataset_NC = datasets.ImageFolder(root=dataroot+train_suffix+nc_suffix, transform=transform)

    # Combine both AD and NC to use for training
    train_dataset = ConcatDataset([train_dataset_AD, train_dataset_NC])

    # Create and return dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

def get_test_loader():
    # Load data
    test_dataset_AD = datasets.ImageFolder(root=dataroot+test_suffix+ad_suffix, transform=transform)
    test_dataset_NC = datasets.ImageFolder(root=dataroot+test_suffix+nc_suffix, transform=transform)

    # Combine both AD and NC to use for training
    test_dataset = ConcatDataset([test_dataset_AD, test_dataset_NC])

    # Create and return dataloader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader





