"""Preprocessing of data, generating and loading of datasets and loaders"""
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import random

image_size = 256
batch_size = 64
crop_size = 192


def get_train_transform():
    """creates the transform used in preprocessing the training and validation data

    Returns:
        torchvision.transforms.functional: training and validation transformation
    """
    transform = transforms.Compose([
        transforms.CenterCrop((crop_size, crop_size)),
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1155], std=[0.2224]) # Calculated values
    ])
    return transform


def get_test_transform():
    """creates the transform used in preprocessing the testing data

    Returns:
        torchvision.transforms.functional: testing transformation
    """
    transform = transforms.Compose([
        transforms.CenterCrop((crop_size, crop_size)),
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1167], std=[0.2228]) # Calculated values
    ])
    return transform


def split_train_data(root, validation_ratio=0.3):
    """completes a patient level split on the training dataset, splitting into training and validation

    Args:
        root (string): the root of the dataset folder
        validation_ratio (float, optional): the percentage of the training data being split into validation data. Defaults to 0.3.

    Returns:
        torch.utils.data.Datasets: the train and validation datasets
    """
    root = os.path.join(root, "train")

    # Load the test data without applying transforms
    dataset = ImageFolder(root=root)

    # Randomly select patient IDs for validation
    random.seed(42)
    patient_ids = list(set(os.path.basename(path).split('_')[0] for path, _ in dataset.samples))

    # Determine the number of patients for validation and test
    num_validation_patients = int(len(patient_ids) * validation_ratio)
    train_patients = set(patient_ids[num_validation_patients:])
    validation_patients = set(patient_ids[:num_validation_patients]) 
    
    # Split test dataset based on patient IDs
    validation_samples = [(path, label) for path, label in dataset.samples if os.path.basename(path).split('_')[0] in validation_patients]
    train_samples = [(path, label) for path, label in dataset.samples if os.path.basename(path).split('_')[0] in train_patients]
    
    train_transform = get_train_transform()

    # Create validation and test datasets
    validation_dataset = ImageFolder(root=root, transform=train_transform)
    train_dataset = ImageFolder(root=root, transform=train_transform)

    # Overwrite samples
    validation_dataset.samples = validation_samples
    train_dataset.samples = train_samples

    return train_dataset, validation_dataset


def load_dataloaders(root):
    """loads the dataloaders for each train, test and validation datasets

    Args:
        root (string): the root of the dataset folder

    Returns:
        torch.utils.data.DataLoader: the train, test and validation dataloaders
    """
    # get transform
    test_transform = get_test_transform()

    # create datasets
    test_dataset = ImageFolder(root + 'test', transform=test_transform)
    train_dataset, validation_dataset = split_train_data(root, validation_ratio=0.3)
    
    # create dataloaders
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_dataloader, test_dataloader, validation_dataloader