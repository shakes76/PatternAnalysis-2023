import os
import random

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from torchvision.datasets import ImageFolder


class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None):
        # Initialize the dataset.
        self.imageFolderDataset = imageFolderDataset
        # Apply the transformations to the images.
        self.transform = transform

    # Get a pair of images from the dataset.
    def __getitem__(self, index):

        # Get a random image from the dataset.
        img0_tuple = self.imageFolderDataset.imgs[index]

        # We need to make sure approx 50% of images are in the same class.
        should_get_same_class = random.randint(0, 1)

        # Get another image from the dataset.
        if should_get_same_class:
            while True:
                # Keep looping till the same class image is found.
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                # Make sure the two images are from the same class.
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # Keep looping till the different class image is found.
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                # Make sure the two images are from the different classes.
                if img0_tuple[1] != img1_tuple[1]:
                    break

        # Load the two images.
        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        # Apply the transforms to the images.
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        # Return the two images and the label (same class or different class).
        return img0, img1, torch.from_numpy(np.array([int(img0_tuple[1] != img1_tuple[1])], dtype=np.float32))

    # Get the number of images in the dataset.
    def __len__(self):
        return len(self.imageFolderDataset.imgs)


# Define the transformations to be applied on the images.
transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert('RGB')),  # Convert grayscale image to RGB
    transforms.RandomResizedCrop(100, scale=(0.8, 1.0), ratio=(1, 1)),  # Randomly crop and resize the image to 100x100, keeping the aspect ratio to 1:1
    transforms.RandomRotation(10),  # Randomly rotate the image by up to 10 degrees
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.Resize((100, 100)),  # Resize the image to 100x100
    transforms.ToTensor()  # Convert the image to a PyTorch tensor
])

# Use relative paths to specify the directories.
train_dir = ImageFolder(root=os.path.join(os.getcwd(), "AD_NC/train"))
test_dir = ImageFolder(root=os.path.join(os.getcwd(), "AD_NC/test"))

# Create the datasets.
train_dataset = SiameseNetworkDataset(imageFolderDataset=train_dir, transform=transform)
test_dataset = SiameseNetworkDataset(imageFolderDataset=test_dir, transform=transform)
