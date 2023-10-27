'''
dataset.py: contains the data loader for loading and preprocessing the data
'''

from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
import torch
import matplotlib.pyplot as plt

'''
Returns a list of unique patient IDs given the dataset path
'''
def get_patient_ids(data_path):
    # Knowing that patient IDs are encoded in image filenames as 'patientID_xx.jpeg'
    all_files = [os.path.basename(f) for dp, dn, filenames in os.walk(data_path) for f in filenames]
    patient_ids = list(set([f.split('_')[0] for f in all_files]))
    return patient_ids

'''
Returns mean and std of the dataset
'''
def compute_mean_std(loader):
    mean = 0.0
    squared_mean = 0.0
    N = 0
    for inputs, _ in loader:
        N += inputs.size(0)
        mean += inputs.sum(dim=(0,2,3))
        squared_mean += (inputs ** 2).sum(dim=(0,2,3))
    mean /= N * inputs.size(2) * inputs.size(3)
    squared_mean /= N * inputs.size(2) * inputs.size(3)
    std = (squared_mean - mean ** 2).sqrt()
    return mean, std

'''
Returns the train, val, and test dataloaders for the AD_NC dataset, with a train/val split of 80/20
'''
def get_alzheimer_dataloader(batch_size:int=32, img_size:int=224, path:str="/content/ADNC-Dataset/AD_NC"):
    train_data_path = path+"/train"
    test_data_path = path+"/test"

    # Transformers WITHOUT normalization (used to compute mean and std)
    pre_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    pre_train_dataset = ImageFolder(root=train_data_path, transform=pre_transforms)

    n_samples = 1000  # Define how many samples you want to use for the estimation
    sampler = SubsetRandomSampler(torch.randperm(len(pre_train_dataset))[:n_samples])
    sample_loader = DataLoader(pre_train_dataset, batch_size=batch_size, sampler=sampler)

    # Compute mean and std for 1000 samples in dataset
    mean, std = compute_mean_std(sample_loader)

    # transformers with normalization as calculated above
    # Random rotations and random sized crop added
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(10),  # Random rotations in range [-10, 10]
        transforms.RandomResizedCrop(224, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # get patient ids from training data path and split these with 80/20
    patient_ids = get_patient_ids(train_data_path)
    train_patient_ids, val_patient_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)

    # create datasets
    train_dataset = ImageFolder(root=train_data_path, transform=train_transforms)
    test_dataset = ImageFolder(root=test_data_path, transform=test_transforms)

    # map ID split to the images to get patient-level split
    train_indices = [i for i, (path, label) in enumerate(train_dataset.samples) if path.split('/')[-1].split('_')[0] in train_patient_ids]
    val_indices = [i for i, (path, label) in enumerate(train_dataset.samples) if path.split('/')[-1].split('_')[0] in val_patient_ids]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    # create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader