from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import random

def dataset_transforms():
    """    
    Transforms to be applied to the training dataset

    Returns:
        (transforms): A compostite of the transforms to be applied
    """    
    return transforms.Compose([transforms.Grayscale(), transforms.Resize((128, 120)), transforms.ToTensor()])

def get_dataloader(dir, batch_size, split):
    """    
    Returns all train, validation and test dataloaders. 
    Dataset split has a set seed so that test dataloader is reproducable for for predict.py

    Args:
        dir (string): directory of dataset
        batch_size (int): batch size for dataloaders
        split (list[float]): Percentages for splitting into train, validate and test datasets
    Returns:
        train_dataloader (DataLoader): dataloader for the train dataset
        valid_dataloader (DataLoader): dataloader for the validation dataset
        test_dataloader  (DataLoader): dataloader for the test dataset
    """    
    dataset = datasets.ImageFolder(root = dir, transform=dataset_transforms())
    train_dataset, valid_dataset, test_dataset = random_split(dataset, split, generator=torch.Generator().manual_seed(46420763))
    
    # Load into SiameseADNIDataset to create pairs
    train_dataset = SiameseADNIDataset(train_dataset)
    valid_dataset = SiameseADNIDataset(valid_dataset)
    test_dataset = SiameseADNIDataset(test_dataset)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader

class SiameseADNIDataset(Dataset):
    """
    ADNI Dataset
    Creates pairs for the Siamese Network.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        """
        Assign a pair for datapoint with given index

        Args:
            index (int): index of item in dataset 

        Returns:
            x1 (tensor): datapoint given by index
            x2 (tensor): randomly assiged pair for x1
            label (float): label of pair
        """
        x1, x1_class = self.dataset[index]
        
        # Get random datapoint from the dataset to form pair
        x2, x2_class = self.dataset[random.randint(0, len(self.dataset) - 1)]
        
        # Label pair
        label = torch.tensor(0.0) if x1_class == x2_class else torch.tensor(1.0)
        return x1, x2, label
    
    def __len__(self):
        return len(self.dataset)