from torchvision.datasets import ImageFolder
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
import random

def get_triplet_train_loader(data_dir, batch_size):
    train_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomCrop(100, padding=4, padding_mode='reflect'),
        transforms.Grayscale(),
    ])
    train_dataset = TripletImageFolder(data_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

def get_triplet_test_loader(data_dir, batch_size):
    test_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Grayscale(),
    ])
    test_dataset = TripletImageTestFolder(data_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return test_loader

def get_triplet_test_loader_predict(data_dir, batch_size):
    test_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Grayscale(),
    ])
    test_dataset = TripletImageTestFolder(data_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def get_datasets(data_dir, transform=None):
    train_dataset = ImageFolder(root=data_dir + '/train', transform=transform)
    test_dataset = ImageFolder(root=data_dir + '/test', transform=transform)
    return train_dataset, test_dataset

def get_classification_dataloader(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Grayscale(),
    ])
    train_dataset = ImageFolder(root=data_dir + '/train', transform=transform)
    test_dataset = ImageFolder(root=data_dir + '/test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    
    
    return train_loader, test_loader

def get_classification_accuracy_dataloader(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Grayscale(),
    ])
    test_dataset = ImageFolder(root=data_dir + '/test', transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)  
    
    return test_loader

def get_dataLoader(data_dir):
    transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Grayscale(),
        ])
    data = ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(data, batch_size=len(data))
    return loader

class TripletImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_folder = datasets.ImageFolder(root, transform=None)
        self.labels = self.image_folder.targets

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        anchor, label1 = self.image_folder[idx]
        while True:
            idx2 = torch.randint(0, len(self), (1,)).item()
            positive, label2 = self.image_folder[idx2]
            if label1 == label2:
                break
        while True:
            idx3 = torch.randint(0, len(self), (1,)).item()
            negative, label3 = self.image_folder[idx3]
            if label1 != label3:
                break
        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        return anchor, positive, negative
    
class TripletImageTestFolder(Dataset):
    def __init__(self, root, transform=None):
        random.seed(42)
        self.root = root
        self.transform = transform
        self.image_folder = datasets.ImageFolder(root, transform=None)
        self.labels = self.image_folder.targets
        self.indexsPoitive = random.sample(range(0, len(self)), len(self))
        self.indexsNegative = random.sample(range(0, len(self)), len(self))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        anchor, label1 = self.image_folder[idx]
        idx2 = 0
        while True:
            if idx + idx2 >= len(self):
                idx = -idx2
            positive, label2 = self.image_folder[self.indexsPoitive[idx + idx2]]
            if label1 == label2:
                break
            idx2 += 1
        idx3 = 0
        while True: 
            if idx + idx3 >= len(self):
                idx = -idx3
            negative, label3 = self.image_folder[self.indexsNegative[idx + idx3]]
            if label1 != label3:
                break
            idx3 += 1
        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        return anchor, positive, negative