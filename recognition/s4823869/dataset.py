'''
Image Dataset Preparation and Visualization

This part of code prepares a dataset from the OASIS dataset for training, including transforming the images and providing data loaders.
It also displays sample images from the dataset.
'''

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt

# Define dataset paths
TRAIN_DATASET_PATH = "/Users/yashmittal/Downloads/Pattern Recognition Project 02/keras_png_slices_data/keras_png_slices_seg_train"  # Training data
TEST_DATASET_PATH = "/Users/yashmittal/Downloads/Pattern Recognition Project 02/keras_png_slices_data/keras_png_slices_seg_test"   # Test data
VALIDATION_DATASET_PATH = "/Users/yashmittal/Downloads/Pattern Recognition Project 02/keras_png_slices_data/keras_png_slices_seg_validate"  # Validation data
BATCH_SIZES = [256, 128, 64, 32, 16, 8]
CHANNELS_IMG = 3

# Customized ImageDataset to read image data
class CustomImageDataset(Dataset):
    def __init__(self, img_dirs, transform=None):
        '''
        Custom Image Dataset

        Args:
            img_dirs (list): List of paths to image directories.
            transform (callable): A function/transform to apply to the images.

        '''
        self.transform = transform
        self.img_files = []
        for dir_path in img_dirs:
            self.img_files += [os.path.join(dir_path, fname) for fname in os.listdir(dir_path)]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        '''
        Get an image from the dataset by index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            torch.Tensor: The image data.

        '''
        img_name = self.img_files[idx]
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

def get_loader(image_size):
    '''
    Create a data loader for the image dataset.

    Args:
        image_size (int): The size to which images should be resized.

    Returns:
        torch.utils.data.DataLoader: Data loader for the image dataset.
        CustomImageDataset: The dataset itself.

    '''
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize the images
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)],
            [0.5 for _ in range(CHANNELS_IMG)],
        )
    ])
    batch_size = BATCH_SIZES[4]  # Image size = 256, batch size = 16

    # Load all the training, test, and validation data together to train the styleGAN model
    dataset = CustomImageDataset(img_dirs=[TRAIN_DATASET_PATH, TEST_DATASET_PATH, VALIDATION_DATASET_PATH], transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dataset

def check_loader():
    '''
    Check and display sample images from the data loader.

    '''
    loader, _ = get_loader(256)
    img = next(iter(loader))
    _, ax = plt.subplots(3, 3, figsize=(8, 8))
    plt.suptitle('Real sample images')
    ind = 0
    for k in range(3):
        for kk in range(3):
            ax[k][kk].imshow((img[ind].permute(1, 2, 0) + 1) / 2)
            ind += 1

    if not os.path.exists("output_images"):

        os.makedirs("output_images")

    # Save the figure to the specified path
    save_path = os.path.join("output_images", "initial_image.png")
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    check_loader()
