from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
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

def train_dataloader(dir, batch_size):
    train_dataset = SiameseADNIDataset(dir + "/train", transform=train_transforms())
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader
    
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
        label = 1 if x1_class == x2_class else 0
        return x1, x2, label
    
    def __len__(self):
        return len(self.dataset)