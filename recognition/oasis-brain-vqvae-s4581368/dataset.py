# Dataset loader for the OASIS brain dataset
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# Data Parameters

test_path = "./keras_png_slices_data/"
train_path = "./keras_png_slices_data/" 
validate_path = "./keras_png_slices_data/"


def data_loaders(train_path, test_path, validate_path, batch_size=32):

    # TODO: Include a composed transform, and split out dataset into train test and validate. For now just
    # batch all data together
    train_dataset = torchvision.datasets.ImageFolder(train_path, transform=transforms.ToTensor())
    train_dataloader = DataLoader(train_dataset[2], batch_size=batch_size, shuffle=True)

    # Test Dataset and DataLoader
    test_dataset = torchvision.datasets.ImageFolder(test_path, transform=transforms.ToTensor())
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Validation Dataset and DataLoader
    validation_dataset = torchvision.datasets.ImageFolder(validate_path)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader

def see_data(data_loader):
    # Display image and label.
    train_features, train_labels = next(iter(data_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    img = torch.transpose(img, 0, 2)
    label = train_labels[0]
    plt.imshow(img)
    plt.show()
    print(f"Label: {label}")

train = data_loaders(train_path, test_path, validate_path)
see_data(train)

