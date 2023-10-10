"""
Contains the data loader for loading and preprocessing the dataset
"""
import torch
import torchvision
import torchvision.transforms as transforms

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Path
# data_path_root = "/s4699048/Datasets/keras_png_slices_data/"  # 336 Workstation Path
data_path_root = "/Datasets/keras_png_slices_data/"

# ----------------
# Data Loader
def generateDataLoader():
    """
    Imports the datasets & creates the dataloaders for the train, test & validation partitions of the dataset.
    Returns: trainset, train_loader, testset, test_loader, validationset, validation_loader
    """
    print("> Loading Dataset")
    # Import Datasets
    # TODO: research which transforms are best to apply to the OASIS dataset (similar to how the MNIST dataset had standard values for normalisation etc)
    trainset = torchvision.datasets.ImageFolder(root=data_path_root+"keras_png_slices_train",
                            transform=transforms.Compose([
                                #    transforms.Resize(image_size),
                                # transforms.Resize((2 ** log_resolution, 2 ** log_resolution)),
                                #    transforms.CenterCrop(image_size),
                                # transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor(),
                                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    
    testset = torchvision.datasets.ImageFolder(root=data_path_root+"keras_png_slices_test",
                            transform=transforms.Compose([
                                #    transforms.Resize(image_size),
                                transforms.Resize((2 ** log_resolution, 2 ** log_resolution)),
                                #    transforms.CenterCrop(image_size),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))

    validationset = torchvision.datasets.ImageFolder(root=data_path_root+"keras_png_slices_validate",
                            transform=transforms.Compose([
                                #    transforms.Resize(image_size),
                                transforms.Resize((2 ** log_resolution, 2 ** log_resolution)),
                                #    transforms.CenterCrop(image_size),
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