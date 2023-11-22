"""
Contains the data loader for loading and preprocessing the dataset
"""
import torch
import torchvision
import torchvision.transforms as transforms
    
# ----------------
# Data Loader
def generateDataLoader(imageHeight: int, imageWidth: int, batch_size: int, data_path_root: str):
    """
    Imports the datasets & creates the dataloaders for the train, test & validation partitions of the dataset.
    Returns: trainset, train_loader, testset, test_loader, validationset, validation_loader
    """
    print("> Loading Dataset")
    # Import Datasets
    trainset = torchvision.datasets.ImageFolder(root=data_path_root+"keras_png_slices_train",
                            transform=transforms.Compose([
                                transforms.Resize((imageHeight, imageWidth)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    
    testset = torchvision.datasets.ImageFolder(root=data_path_root+"keras_png_slices_test",
                            transform=transforms.Compose([
                                transforms.Resize((imageHeight, imageWidth)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))

    validationset = torchvision.datasets.ImageFolder(root=data_path_root+"keras_png_slices_validate",
                            transform=transforms.Compose([
                                transforms.Resize((imageHeight, imageWidth)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))

    # Construct Data Loaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)  # Data Loader
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)  # Data Loader
    validation_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)  # Data Loader
    print("> Dataset Ready")

    return trainset, train_loader, testset, test_loader, validationset, validation_loader