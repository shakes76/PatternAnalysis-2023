"""
LOADING AND TRANSFORMING THE DATA
"""

# Imports
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

# Data Loading Parameters

# if training on rangpur, following parameters need to be
# dataroot = "/home/groups/comp3710/ADNI/AD_NC"
# test_suffix = "/test"
# train_suffix = "/train"

# these parameters are for if the data is in the project root
dataroot = ".\\ADNI\\AD_NC"
test_suffix = "\\test"
train_suffix = "\\train"
batch_size = 128

# Create Data Transforms
# Train transforms added to hopefully reduce overfit to training data
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToTensor()
])

# Function to get dataloader from other files
def get_train_loader():
    # Load data
    train_dataset= datasets.ImageFolder(root=dataroot+train_suffix, transform=train_transform)

    # Create and return dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

def get_test_loader():
    # Load data
    test_dataset = datasets.ImageFolder(root=dataroot+test_suffix, transform=test_transform)

    # Create and return dataloader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


# Function to Downscale images for input
original_size = (240, 256) # original image size
def downscale(images, factor = 2):
    return transforms.Resize(tuple(map(lambda x: x//factor, original_size)), antialias=True)(images)