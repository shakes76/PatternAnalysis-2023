from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import random


def get_transform(image_size, crop_size):
    transform = transforms.Compose([
        transforms.CenterCrop((crop_size, crop_size)),
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    return transform


def split_test_data(root, image_size, crop_size, test_val_ratio=0.5):
    test_dir = os.path.join(root, "test")

    # Load the test data without applying transforms
    test_data = ImageFolder(root=test_dir)

    # Extract patient IDs
    patient_ids = set(os.path.basename(path).split('_')[0] for path, _ in test_data.samples)

    # Determine the number of patients for validation and test
    num_validation_patients = int(len(patient_ids) * test_val_ratio)

    # Randomly select patient IDs for validation
    random.seed(42)
    patient_ids = list(set(os.path.basename(path).split('_')[0] for path, _ in test_data.samples))
    
    # Split test dataset based on patient IDs
    validation_samples = [(path, label) for path, label in test_data.samples if os.path.basename(path).split('_')[0] in patient_ids]
    test_samples = [(path, label) for path, label in test_data.samples if os.path.basename(path).split('_')[0] not in patient_ids]
    
    transform = get_transform(image_size, crop_size)

    # Create validation and test datasets
    validation_dataset = ImageFolder(root=test_dir, transform=transform)
    test_dataset = ImageFolder(root=test_dir, transform=transform)

    # Overwrite samples
    validation_dataset.samples = validation_samples
    test_dataset.samples = test_samples

    return validation_dataset, test_dataset


def load_dataset(image_size, crop_size, batch_size, root):    
    # transform    
    transform = get_transform(image_size, crop_size)

    # create datasets
    train_dataset = ImageFolder(root + 'train', transform=transform)
    validation_dataset, test_dataset = split_test_data(root, image_size, crop_size, test_val_ratio=0.5)
    
    # create dataloaders
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_dataloader, test_dataloader, validation_dataloader
