import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from utils import *
import math

"""
Generate data loaders to be used for training and predicting.
"""
transform = transforms.Compose([
                            transforms.Grayscale(1),
                            transforms.ToTensor()
                        ])

train_dataset = ImageFolder(root=train_dir, transform=transform)
    
train_subset = Subset(train_dataset, range(math.floor(len(train_dataset) * 0.75)))
validate_sbuset = Subset(train_dataset, range(math.floor(len(train_dataset) * 0.75), len(train_dataset)))

def generate_train_loader():
    """
    Generate a Dataloader consisting of the ADNI training data as 
    grayscale tensors.

    Returns:
        A dataloader with ADNI traninig data, both AD and NC. 
        Batch size as specified in utils.py.

    """
    
    return DataLoader(train_subset, batch_size=batch_size, shuffle=True)

def generate_validation_loader():
    """
    Generate a Dataloader consisting of the ADNI validation data as 
    grayscale tensors.

    Returns:
        A dataloader with ADNI traninig data, both AD and NC. 
        Batch size as specified in utils.py.

    """
    
    return DataLoader(validate_sbuset, batch_size=batch_size, shuffle=True)

def generate_test_loader():
    """
    Generate a Dataloader consisting of the ADNI testing data as grayscale 
    tensors.

    Returns:
        A dataloader with ADNI testing data, both AD and NC. Batch size as 
        specified in utils.py.

    """
    
    test_dataset = ImageFolder(root=test_dir, transform=transform)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
