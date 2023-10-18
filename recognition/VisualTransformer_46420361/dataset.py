"""Preprocessing of data, generating and loading of datasets and loaders"""
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import random
import cv2
import numpy as np
from PIL import Image


class CropBrainScan:
    def __call__(self, image):
        # Convert the image to a NumPy array if it's not already
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Ensure the image is in the CV_8UC1 format
        if image.ndim > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to separate the brain scan from the background
        _, thresholded = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour (the brain scan region)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the coordinates of the bounding box around the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop the image to keep only the brain scan region
        cropped_image = image[y:y + h, x:x + w]

        # Convert the NumPy array back to a PIL image
        cropped_image = Image.fromarray(cropped_image)
        
        return cropped_image


def get_train_transform(image_size, crop_size):
    transform = transforms.Compose([
        # CropBrainScan(),
        transforms.CenterCrop((crop_size, crop_size)),
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1155], std=[0.2224]) # Calculated values
    ])
    return transform


def get_test_transform(image_size, crop_size):
    transform = transforms.Compose([
        # CropBrainScan(),
        transforms.CenterCrop((crop_size, crop_size)),
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1167], std=[0.2228]) # Calculated values
    ])
    return transform


# def split_test_data(root, image_size, crop_size, test_val_ratio=0.5):
#     test_dir = os.path.join(root, "test")

#     # Load the test data without applying transforms
#     test_data = ImageFolder(root=test_dir)

#     # Randomly select patient IDs for validation
#     random.seed(42)
#     patient_ids = list(set(os.path.basename(path).split('_')[0] for path, _ in test_data.samples))

#     # Determine the number of patients for validation and test
#     num_patients = int(len(patient_ids) * test_val_ratio)
#     test_patients = set(patient_ids[num_patients:])
#     validation_patients = set(patient_ids[:num_patients]) 
    
#     # Split test dataset based on patient IDs
#     validation_samples = [(path, label) for path, label in test_data.samples if os.path.basename(path).split('_')[0] in validation_patients]
#     test_samples = [(path, label) for path, label in test_data.samples if os.path.basename(path).split('_')[0] in test_patients]
    
#     train_transform = get_train_transform(image_size, crop_size)
#     test_transform = get_test_transform(image_size, crop_size)

#     # Create validation and test datasets
#     validation_dataset = ImageFolder(root=test_dir, transform=train_transform)
#     test_dataset = ImageFolder(root=test_dir, transform=test_transform)

#     # Overwrite samples
#     validation_dataset.samples = validation_samples
#     test_dataset.samples = test_samples

#     return validation_dataset, test_dataset


def load_dataloaders(root, image_size, crop_size, batch_size):
    # transform    
    train_transform = get_train_transform(image_size, crop_size)
    test_transform = get_test_transform(image_size, crop_size)

    # create datasets
    train_dataset = ImageFolder(root + 'train', transform=train_transform)
    test_dataset = ImageFolder(root + 'test', transform=test_transform)
    # validation_dataset, test_dataset = split_test_data(root, image_size, crop_size, batch_size, test_val_ratio=0.5)
    
    # create dataloaders
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    # validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_dataloader, test_dataloader
