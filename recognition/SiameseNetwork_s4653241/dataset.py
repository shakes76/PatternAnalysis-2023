# Importing necessary libraries and modules
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,ConcatDataset,Dataset,TensorDataset, Subset
from PIL import Image
import random

def get_transform():
    """
        Returns a composed transform for preprocessing images.
    Returns:
        torchvision.transforms.Compose: A composed transform for preprocessing images.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale with one channel
        transforms.Resize((120,128)),  # Resize to (120,128)
        transforms.ToTensor(),
        # You can add more transformations if needed
    ])
    return transform

def get_dataloader(dataset, batch_size = 16, shuffle=True):
    """
    Returns a DataLoader for the given dataset.

    This function creates and returns a DataLoader for the provided dataset with the specified batch size and shuffling option.

    Parameters:
    - dataset (Dataset): The dataset for which the DataLoader is to be created.
    - batch_size (int, optional): The number of samples per batch. Default is 16.
    - shuffle (bool, optional): Whether to shuffle the dataset before splitting into batches. Default is True.

    Returns:
    - DataLoader: A DataLoader object for the given dataset with the specified parameters.
    """
    v_dataloader = DataLoader(dataset,
                            shuffle=shuffle,
                            num_workers=1,
                            batch_size=batch_size)
    return v_dataloader



def visualise_1(dataset):
    """
    Plots a random image in the dataset

    Args:
        dataset (Dataset): Dataset which the random image will be picked
    """
    img,lab = random.choice(dataset)
    plt.title(lab)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig("visualise_1")
    plt.show()

    
def visualise_batch(dataloader):
    """
    Plots a batch of images from the dataloader

    Args:
        dataloader (Dataloader): Dataset which a batch will be taken to be plotted.
    """
    LABELS = ['POS','NEG']

    example_batch = iter(dataloader)
    images1,images2,labels = next(example_batch)

    plt.figure(figsize=(16,4)) # width x height
    batch_size = len(images1)
    for idx in range(batch_size):

        image1 = transforms.ToPILImage()(images1[idx])
        image2 = transforms.ToPILImage()(images2[idx])
        label = LABELS[int(labels[idx].item())]

        plt.subplot(2,batch_size,idx+1)
        
        plt.imshow(image1,cmap='gray')
        plt.axis('off')

        plt.subplot(2,batch_size,idx+1+batch_size)
        plt.imshow(image2,cmap='gray')
        plt.title(label)
        plt.axis('off')

    plt.savefig("visualise_batch")
    plt.show()



class SiameseNetworkDataset1(Dataset):
    """    
    A dataset class for creating pairs of images for Siamese networks.

    Args:
        Dataset (torchvision.datasets.ImageFolder): A dataset object containing images and their labels.
        transform (torchvision.transforms): A function/transform that takes in an image and returns a transformed version. 
                                        Default is None.
    """
    def __init__(self,imageFolderDataset,transform=None):
        
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self,index):
        """
        Returns a pair of images and a label indicating if they belong to the same class.

        Args:
            index (int): Index (ignored)

        Returns:
            tuple:  A tuple containing two images and a label
        """
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        #We need to approximately 50% of images to be in the same class
        should_get_same_class = random.randint(0,1)
        if should_get_same_class:
            while True:
                #Look untill the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:

            while True:
                #Look untill a different class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


class SiameseNetworkDataset_test(Dataset):
    """    
    A test dataset class for creating pairs of images for Siamese networks.

    Args:
        Dataset (torchvision.datasets.ImageFolder): A dataset object containing images and their labels.
        transform (torchvision.transforms): A function/transform that takes in an image and returns a transformed version. 
                                        Default is None.
    """
    def __init__(self,imageFolderDataset,transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self,index):
        """
        Returns a pair of images, a label indicating if they belong to the same class and labels of images.

        Args:
            index (int): Index (ignored in this implementation as images are chosen randomly).

        Returns:
            tuple: A tuple containing two images, a label (1 if the images are from different classes, 0 otherwise) and 2 labels of the respective images.
        """
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        #We need to approximately 50% of images to be in the same class
        should_get_same_class = random.randint(0,1)
        if should_get_same_class:
            while True:
                #Look untill the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:

            while True:
                #Look untill a different class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32)), img0_tuple[1],img1_tuple[1]

    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
            int: Total number of images in the dataset.
        """
        return len(self.imageFolderDataset.imgs)
