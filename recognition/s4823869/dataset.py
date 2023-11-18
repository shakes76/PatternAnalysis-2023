"""
Data Loading for Progressive Growing GAN

This script loads image data for training a Progressive Growing Style GAN (PGGAN). It utilizes the CustomImageData
class to organize and preprocess the dataset for training. The script also includes a function to display sample
images from the dataset.

@author: Yash Mittal
@ID: s48238690
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt

# Define dataset paths
TRAIN_DATA_PATH = "/Users/yashmittal/Downloads/Pattern Recognition Project 02/keras_png_slices_data/keras_png_slices_seg_train"  # Training data
TEST_DATA_PATH = "/Users/yashmittal/Downloads/Pattern Recognition Project 02/keras_png_slices_data/keras_png_slices_seg_test"  # Test data
VALIDATION_DATA_PATH = "/Users/yashmittal/Downloads/Pattern Recognition Project 02/keras_png_slices_data/keras_png_slices_seg_validate"  # Validation data
CHANNELS_IMG = 3


class CustomImageData(Dataset):
    """
    A custom PyTorch dataset class to load and preprocess image data for training.

    Args:
        data_dirs (list): List of directory paths containing image data.
        transform (callable): A function/transform to apply to the images.

    Attributes:
        transform (callable): Image transformation function.
        image_files (list): List of image file paths.

    Methods:
        __len__(): Returns the number of images in the dataset.
        __getitem__(idx): Loads and processes an image from the dataset by index.
    """

    def __init__(self, data_dirs, transform=None):
        self.transform = transform
        self.image_files = []
        for dir_path in data_dirs:
            self.image_files += [os.path.join(dir_path, fname) for fname in os.listdir(dir_path)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


def get_data_loader(image_size):
    """
    Create a data loader for image data to be used for training.

    Args:
        image_size (int): Size to which images are resized.

    Returns:
        DataLoader: A PyTorch DataLoader for training data.
        Dataset: The custom dataset object containing image data.
    """

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)],
            [0.5 for _ in range(CHANNELS_IMG)],
        )
    ])

    dataset = CustomImageData(data_dirs=[TRAIN_DATA_PATH, TEST_DATA_PATH, VALIDATION_DATA_PATH], transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    return loader, dataset


def display_sample_images():
    """
    Display a grid of sample images from the dataset.
    """
    loader, _ = get_data_loader(256)
    images = next(iter(loader))
    _, ax = plt.subplots(8, 8, figsize=(8, 8))
    plt.suptitle('Original images')
    index = 0
    for k in range(8):
        for kk in range(8):
            ax[k][kk].imshow((images[index].permute(1, 2, 0) + 1) / 2)
            index += 1

    if not os.path.exists("output_images"):
        os.makedirs("output_images")

    save_path = os.path.join("output_images", "initial_input_image.png")
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    display_sample_images()
