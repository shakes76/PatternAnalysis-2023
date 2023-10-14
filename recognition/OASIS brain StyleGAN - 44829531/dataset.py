import os
import torch
from torchvision import datasets, transforms

local_dataset_path = "C:\\Users\\ethan\\Desktop\\COMP3710\\keras_png_slices_train"
# dataset_path = "/home/groups/comp3710/OASIS/keras_png_slices_train"
dataset_path = local_dataset_path


def load_data(dataset_path):
    """
    Load the dataset from the given path.

    param: dataset_path: The path to the dataset
    return: The dataset of images
    """
    file_path = os.path.join(os.getcwd(), dataset_path)

    # Convert to grayscale and convert to tensor and normalise to [0, 1]
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    # Load the dataset
    image_data = datasets.ImageFolder(root=file_path, transform=transform)
    return image_data
