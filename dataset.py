import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data
import os

def get_dataloaders(batch_size: int, workers: int, image_resize: int, dataroot: str, rgb: bool):
    """Returns a training dataloader, test dataloader, and validation dataloader.
    Assumes that dataroot contains 'train' and 'test' folders.
    """
    # Get file paths
    train_dataroot = os.path.join(dataroot, "train")
    test_dataroot = os.path.join(dataroot, "test")

    # Convert to grayscale if necessary
    if not rgb:
        transform = transforms.Compose([
                                    transforms.Resize((image_resize, image_resize)),
                                    transforms.ToTensor(),
                                    transforms.Grayscale()
                                ])
    else:
        transform = transforms.Compose([
                                    transforms.Resize((image_resize, image_resize)),
                                    transforms.ToTensor(),
                                ])
    
    # Initiate datasets and apply transformations
    full_train_dataset = dset.ImageFolder(root=train_dataroot, transform=transform)
    test_dataset = dset.ImageFolder(root=test_dataroot, transform=transform)

    # Split trainig dataset into (80/20) training/validation sets
    train_size = int(0.8 * len(full_train_dataset))
    validation_size = len(full_train_dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, validation_size])

    # Create the dataloader for each dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=workers)

    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size,
                                                    shuffle=False, num_workers=workers)
    
    return train_loader, test_dataloader, validation_loader