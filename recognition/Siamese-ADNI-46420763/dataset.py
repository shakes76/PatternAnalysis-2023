from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import random

def train_transforms():
    """
    Transforms to be applied to the training dataset
    """
    return transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize(0.1155, 0.2254)])

def test_transforms():
    """
    Transforms to be applied to the test dataset
    """
    return transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize(0.1155, 0.2254)])

def train_dataloader(dir, batch_size, validation_split):
    dataset = SiameseADNIDataset(dir + "/train", transform=train_transforms())
    train_dataset, valid_dataset = random_split(dataset, [1 - validation_split, validation_split])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, valid_dataloader
    
def test_dataloader(dir, batch_size):
    test_dataset = SiameseADNIDataset(dir + "/test", transform=test_transforms())
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    return test_dataloader

class SiameseADNIDataset(Dataset):
    """
    ADNI Dataset for Siamese Network
    """
    def __init__(self, dir, transform):
        self.dataset = datasets.ImageFolder(root = dir, transform=transform)
        
    def __getitem__(self, index):
        x1, x1_class = self.dataset[index]
        # Get random datapoint from the dataset
        x2, x2_class = self.dataset[random.randint(0, len(self.dataset) - 1)]
        label = torch.tensor(0.0) if x1_class == x2_class else torch.tensor(1.0)
        return x1, x2, label
    
    def __len__(self):
        return len(self.dataset)