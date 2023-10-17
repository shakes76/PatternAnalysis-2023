"""
Author: Rohan Kollambalath
Student Number: 46963765
COMP3710 Sem2, 2023.
Data loader for the ADNI dataset. 
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

class ADNI_Dataset:
    """Builder class for the ADNI dataset. Creates a dataset with specified batch size"""
    def __init__(self, batch_size=32):
        # define the batch size and image locations
        self.batch_size = batch_size
        self._train_root = "./recognition/alzheimers_transformer_46963765/data/train"
        self._test_root = "./recognition/alzheimers_transformer_46963765/data/test"
    
    # To get the training dataloader
    def get_train_and_valid_loader(self, location=None, transform=None):
        """
        Takes the training images and splits into testing and validation sets. 21000 images in folder 
        split into 190000 for training and 2000 for validation. Extra precautions taken to prevent 
        data leakage. Images sorted by patient ID to make sure no one patient is in both training and 
        validation.
        """
        #can provide seperate locations and transformations if needed, else default
        if location != None:
            root_path = location
        else:
            root_path = self._train_root
        if transform == None:
            transform = self.get_transformation("train")
            
        # transform and load in dataloader with shuffle
        train_dataset = torchvision.datasets.ImageFolder(root=root_path, transform=transform)
        
        train_dataset.samples = sorted(train_dataset.samples, key=lambda x: os.path.basename(x[0]))
        ad_count = 0
        nc_count = 0

        # Initialize lists to store the selected indices for each class
        ad_indices = []
        nc_indices = []

        # Iterate through the sorted dataset and select 1000 images for each class
        for idx, (image_path, class_index) in enumerate(train_dataset.samples):
            if class_index == 0 and ad_count < 1000: # for AD
                ad_indices.append(idx)
                ad_count += 1
            elif class_index == 1 and nc_count < 1000: # for NC
                nc_indices.append(idx)
                nc_count += 1

        selected_indices = ad_indices + nc_indices
        # split off validation set based off sorted indices to ensure all images of each patient are taken
        validation_set = torch.utils.data.Subset(train_dataset, selected_indices)
        train_dataset.samples = [train_dataset.samples[i] for i in range(len(train_dataset.samples)) if i not in selected_indices]

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_loader = DataLoader(validation_set, batch_size=self.batch_size, shuffle=True)

        return train_loader, validation_loader 

    # To get the testing dataloeader
    def get_test_loader(self, location=None, transform=None):
        """Method to take test images and return loader"""
        #can provide seperate locations and transformations if needed, else default
        if location != None:
            root_path = location
        else:
            root_path = self._test_root    
        if transform == None:
            transform = self.get_transformation("test")
            
        # transform and load in dataloader with shuffle
        test_dataset = torchvision.datasets.ImageFolder(root=root_path, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        return test_loader 
    
    # image starts off at 3x240x256 need to convert to 1x240x240
    def get_transformation(self, type):
        """Method that dictates the transformations done in pre processing. Includes cropping 
           image to event size. Blur, random rotation and changing to single dimention 
           on top of custom normalisation.
        """
        # apply crops, grayscale and normalisation
        if type == "train":
            transform_method = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=15),
            transforms.GaussianBlur(kernel_size=3),  # Apply Gaussian blur with a specified kernel size
            transforms.RandomCrop(240),
            transforms.Grayscale(num_output_channels=1),
            transforms.Normalize((0.1232,), (0.2308,))
            ])
        else:
            transform_method = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(240),
            transforms.Grayscale(num_output_channels=1),
            transforms.Normalize((0.1232,), (0.2308,))
            ])
            
        return transform_method
    
    

class Model_Visualiser:
    """ Class to visualise data loaders in different ways"""
    
    def __init__(self, loader) -> None:
        """Instantiates the visualiser with a data loader"""
        # set the dataloader being visualised
        self._loader = loader
    
    #method to visialise the images contianed by class
    def visualise(self):
        """Method to visualise nine brains in the dataloader with matplotlib"""
        displayed_count = 0
        rows, cols = 2, 5  # Set the number of rows and columns for the grid

        # Create a new figure
        fig = plt.figure(figsize=(12, 6))

        for batch in self._loader:
            images, labels = batch

            # Iterate through the images
            for i, image in enumerate(images):
                if displayed_count >= 9:
                    break

                # Create a subplot
                ax = plt.subplot(rows, cols, displayed_count + 1)
                ax.set_title(f"{displayed_count}")
                ax.axis('off')

                # Display the image
                plt.imshow(image.permute(1, 2, 0))
                displayed_count += 1
            break
        # Adjust layout and display the figure
        plt.tight_layout()
        plt.show()
            
    # Method used to calculate mean and stf for the dataset previously used for normalisation
    def compute_mean_and_std_for_images(self):
        """
        Method to iterate through the dataloader and calculate the mean and standard
        deveation of the dataset for the purpose of normalisation.
        """
        mean = 0.
        std = 0.
        total_samples = 0

        # Go through the dataloader and calculate the total mean and std
        for data, _ in self._loader:
            batch_samples = data.size(0)
            channels = data.size(1)
            data = data.view(batch_samples, channels, -1)

            # Calculate mean and std for each channel (RGB)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            total_samples += batch_samples

        # divide by total samples
        mean /= total_samples
        std /= total_samples
        return mean, std

