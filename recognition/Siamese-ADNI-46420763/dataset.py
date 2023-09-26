from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import random

def dataset_transforms():
    """
    Transforms to be applied to the training dataset
    """
    return transforms.Compose([transforms.Grayscale(), transforms.Resize((128, 120)), transforms.ToTensor()])

def get_dataloader(dir, batch_size, split):
    dataset = datasets.ImageFolder(root = dir, transform=dataset_transforms())
    train_dataset, valid_dataset, test_dataset = random_split(dataset, split, generator=torch.Generator().manual_seed(46420763))
    train_dataset = SiameseADNIDataset(train_dataset)
    valid_dataset = SiameseADNIDataset(valid_dataset)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader

class SiameseADNIDataset(Dataset):
    """
    ADNI Dataset for Siamese Network
    """
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        x1, x1_class = self.dataset[index]
        # Get random datapoint from the dataset
        x2, x2_class = self.dataset[random.randint(0, len(self.dataset) - 1)]
        label = torch.tensor(0.0) if x1_class == x2_class else torch.tensor(1.0)
        return x1, x2, label
    
    def __len__(self):
        return len(self.dataset)