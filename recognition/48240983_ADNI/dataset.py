"""
Created on Wednesday October 18
Alzheimer's disease using PyTorch (ViT Transformer)
This code defines a custom dataset (CustomDataset) for a computer vision task using PyTorch.
It loads images from subdirectories, applies specified data transformations, and stores the image paths 
along with their corresponding class labels. 

@author: Gaurika Diwan
@ID: s48240983
"""
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Initialised the CustomDataset constructor.

        Args:
            data_dir (str): Directory path containing the dataset.
            transform (callable, optional): A function to apply to the images. Default taken  None.
        """

        self.data_dir = data_dir
        self.transform = transform

        # Given lists to store image paths and labels

        self.image_paths = []
        self.labels = []

        # Subdirectories in the data directory

        classes = os.listdir(data_dir)

        # Loop in classes and collect image paths and labels
        for class_id, class_name in enumerate(classes):
            class_path = os.path.join(data_dir, class_name)
            image_files = os.listdir(class_path)
            self.image_paths.extend([os.path.join(class_path, img) for img in image_files])
            self.labels.extend([class_id] * len(image_files))

    def __len__(self):
        """
        Take the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get an item (image and label) from the dataset.

        Args:
            idx (int): Index of the item to fetch.

        Returns:
            tuple: A tuple containing the image (Tensor) and its respective label (int).
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label

#  Data directory
data_dir = './recognition/48240983_ADNI/AD_NC/train'

#  Data transformations
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])

# Instance of your custom dataset
custom_dataset = CustomDataset(data_dir, transform=transform)

# Data loader for batch processing
batch_size = 16
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

#  Iterate through the data_loader in training script
for images, labels in data_loader:
    
    pass
