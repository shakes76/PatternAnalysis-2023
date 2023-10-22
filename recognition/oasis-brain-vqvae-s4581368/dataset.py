"""
Dataset loader for the OASIS brain dataset
45813685
Ryan Ward
"""
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from skimage import io

class OASISDataSet(Dataset):
    def __init__(self, data_path):
        """
        Initialises the data loader
        :param str data_path: The filepath of the desired batch of images
        """
        self.path = data_path
        self.data = os.listdir(self.path)
        # Only a simple transform, always convert to tensor
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        """
        Returns the number of images in the dataset
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Allows an image to be indexed from the dataset
        :param int index: The index of the desired image
        """
        image_path = os.path.join(self.path, self.data[index])
        image = io.imread(image_path)
        image = self.transform(image)
        return image

def data_loaders(train_path, test_path, validate_path, batch_size=32):
    """
    Generate the dataloaders for the OASIS train, test and validation sets
    :param str train_path: The filepath for the train path directory
    :param str test_path: The filepath for the test path directory
    :param str validate_path: The filepath for the validation path directory
    :returns Dataloader, Dataloader, Dataloader: The respective dataloaders for each
            path in the input
    """
   
    # Train Dataset and Dataloader
    train_dataset = OASISDataSet(train_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Test Dataset and DataLoader
    test_dataset = OASISDataSet(test_path)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # Validation Dataset and DataLoader
    validation_dataset = OASISDataSet(validate_path)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    
    return train_dataloader, test_dataloader, validation_dataloader
    
def see_data(data_loader):
    """
    Helper function to view the loaded data
    :param Dataloader data_loader: The dataloader to view images from
    """
    train_features, train_labels = next(iter(data_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    img = torch.transpose(img, 0, 2)
    label = train_labels[0]
    plt.imshow(img)
    plt.show()
    print(f"Label: {label}")

