import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import PIL.ImageOps
import torch

# Define a custom dataset class that inherits from PyTorch's Dataset class
class CustomDataset(Dataset):

    def __init__(self, image_dataset, transform=None, should_invert=True):
        # Initialize the custom dataset with the image dataset, transformation, and inversion flag
        self.image_dataset = image_dataset
        self.transform = transform
        self.should_invert = should_invert

    # Custom method to get items (pairs of images and a label)
    def __getitem__(self, index):
        # Choose a random image (img1) and decide whether to get a positive or negative pair
        img1_tuple = random.choice(self.image_dataset.imgs)
        should_get_same_class = random.randint(0, 1)
        
        # If it's a positive pair, choose another image (img2) from the same class
        if should_get_same_class:
            while True:
                img2_tuple = random.choice(self.image_dataset.imgs)
                if img1_tuple[1] == img2_tuple[1]:
                    break
        # If it's a negative pair, choose an image (img2) from a different class
        else:
            while True:
                img2_tuple = random.choice(self.image_dataset.imgs)
                if img1_tuple[1] != img2_tuple[1]:
                    break

        # Open the images and convert them to grayscale
        img1 = Image.open(img1_tuple[0])
        img2 = Image.open(img2_tuple[0])
        img1 = img1.convert("L")
        img2 = img2.convert("L")

        # If should_invert is True, invert the pixel values in the images
        if self.should_invert:
            img1 = PIL.ImageOps.invert(img1)
            img2 = PIL.ImageOps.invert(img2)

        # If a transformation is provided, apply it to both images
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # Create a label (1 for positive pair, 0 for negative pair) as a tensor
        label = torch.from_numpy(np.array([int(img1_tuple[1] != img2_tuple[1])], dtype=np.float32))

        # Return img1, img2, and the label
        return img1, img2, label

    # Custom method to get the length of the dataset
    def __len__(self):
        return len(self.image_dataset.imgs)
