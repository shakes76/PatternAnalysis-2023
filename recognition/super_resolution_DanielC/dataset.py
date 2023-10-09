import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from utils import *

def generate_train_loader():
    transform = transforms.Compose([
                            transforms.Resize((60,64)),
                            transforms.ToTensor(),
                            transforms.RandomHorizontalFlip()
                        ])

    AD_dataset = ImageFolder(root=AD_train_dir, transform=transform)
    NC_dataset = ImageFolder(root=NC_train_dir, transform=transform)

    return DataLoader(ConcatDataset([AD_dataset, NC_dataset]), batch_size=batch_size, shuffle=True)

def generate_test_loader():
    transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.RandomHorizontalFlip()
                        ])
    
    AD_dataset = ImageFolder(root=AD_test_dir, transform=transform)
    NC_dataset = ImageFolder(root=NC_test_dir, transform=transform)

    return DataLoader(ConcatDataset([AD_dataset, NC_dataset]), batch_size=batch_size, shuffle=True)
