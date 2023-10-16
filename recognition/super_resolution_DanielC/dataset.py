import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from utils import *

def generate_train_loader():
    """
    Generate a Dataloader consisting of the ADNI training data as grayscale tensors.

    Returns:
        A dataloader with ADNI traninig data, both AD and NC. Batch size as specified in utils.py.

    """
    transform = transforms.Compose([
                            transforms.Grayscale(1),
                            transforms.ToTensor()
                        ])

    train_dataset = ImageFolder(root=train_dir, transform=transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def generate_test_loader():
    """
    Generate a Dataloader consisting of the ADNI testing data as grayscale tensors.

    Returns:
        A dataloader with ADNI testing data, both AD and NC. Batch size as specified in utils.py.

    """
    transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.RandomHorizontalFlip()
                        ])
    
    test_dataset = ImageFolder(root=test_dir, transform=transform)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
