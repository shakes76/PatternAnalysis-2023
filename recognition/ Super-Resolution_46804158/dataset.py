import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms.functional as fn


class ADNIDataset():
    def __init__(self):
        """
        Contains the data loader for loading and preprocessing the MRI data.
        """
        train_data_path = '/Users/mj/Documents/COMP3710_code/COMP3710_Report/AD_NC/train'
        test_data_path = '/Users/mj/Documents/COMP3710_code/COMP3710_Report/AD_NC/test'

        # Define data transformation an preprocess data
        transform = transforms.Compose([
            transforms.CenterCrop(240),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Apply transformations to datasets
        train_dataset = ImageFolder(root=train_data_path, transform=transform)
        test_dataset = ImageFolder(root=test_data_path, transform=transform)

        self.train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)


data = ADNIDataset()
train_img, train_label = next(iter(data.train_loader))
print(train_img.shape)
print(train_label)

test_img, test_label = next(iter(data.test_loader))
print(test_img.shape)
print(test_label)