from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import torch
import random

def train_dataset_transforms():
    """    
    Transforms to be applied to the training dataset

    Returns:
        (transforms): A compostite of the transforms to be applied
    """    
    return transforms.Compose([transforms.Grayscale(), 
                               transforms.RandomRotation(30), 
                               transforms.RandomCrop((256, 240), padding = 16, padding_mode='reflect'),
                               transforms.ToTensor()])

def test_dataset_transforms():
    """    
    Transforms to be applied to the training dataset

    Returns:
        (transforms): A compostite of the transforms to be applied
    """    
    return transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

def get_dataloader(dir, batch_size, train_split):
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
    train_dataset = datasets.ImageFolder(root = dir + '/train', transform=train_dataset_transforms())
    train_dataset, valid_dataset = patient_split(train_dataset, train_split)
    test_dataset = datasets.ImageFolder(root = dir + '/test', transform=test_dataset_transforms())
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader

def patient_split(dataset, train_split):
    dataset_length = len(dataset)//20 # 20 samples per patient
    dataset_indices = list(range(dataset_length))
    random.shuffle(dataset_indices)
    
    train_len = int(dataset_length * train_split)
    
    train_indices = dataset_indices[:train_len]
    valid_indices = dataset_indices[train_len:]
    
    train_dataset = Subset(dataset, [index * 20 + j for index in train_indices for j in range(20)])
    valid_dataset = Subset(dataset, [index * 20 + j for index in valid_indices for j in range(20)])
    return train_dataset, valid_dataset