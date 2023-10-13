'''
contains the data loader for loading and preprocessing the data
'''

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

def get_alzheimer_dataloader():
    # Paths to your training and test datasets
    train_data_path = "./dataset/AD_NC/train"
    test_data_path = "./dataset/AD_NC/test"

    # Transformers 
    # Currently using some standard augmentations for ViTs
    train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to a consistent size
    transforms.RandomHorizontalFlip(),  # Randomly flip horizontally
    transforms.RandomRotation(degrees=15),  # Random rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])

    test_transforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    # Create datasets
    train_dataset = ImageFolder(root=train_data_path, transform=train_transforms)
    test_dataset = ImageFolder(root=test_data_path, transform=test_transforms)

    # Create workers
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers = 4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers = 4)

    return train_loader, test_loader