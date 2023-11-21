'''
Loads the data and preprocesses it for use in the model.

Author: Sophie Bates, s4583766.
'''

import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.utils import save_image
from PIL import Image

class OasisDataset(Dataset):
    """OASIS dataset.
    
    Attributes
    ----------
    data_path : str
        Path to the dataset.
    images : list
        List of image names.
    transform : torchvision.transforms
        Transformations to apply to the images.

    Methods
    -------
    __len__()
        Returns the length of the dataset.
    __getitem__(x)
        Returns the image at index x.

    Ref: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.images = os.listdir(data_path)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, x):
        img_path = self.data_path + self.images[x]
        image = Image.open(img_path)

        image = self.transform(image)
        return image

def show_batch(dl, filename):
    """Plot images grid of single batch"""
    imgs = next(iter(dl))
    img = make_grid(imgs[0])
    show_images(img)
    save_image(img, filename)

def show_images(img, epoch):
    """Plot images grid of single batch"""
    img = img.numpy()
    fig = plt.imshow(np.transpose(img, (1,2,0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    
    # Save to 'gen_img' subdirectory
    img_name = f's_epoch_{epoch}.png'
    save_image(img, img_name)
    print("Saving", img_name)
    plt.savefig(img_name)
    plt.clf()

def get_dataloaders(data_path_training, data_path_testing, data_path_validation, batch_size):
    """Load the dataset and returns the three data loaders.

    Parameters
    ----------
    The three paths to the training, testing and validation datasets, and batch size.

    Returns
    -------
    The DataLoaders for the training, testing and validation datasets.
    """
    train_ds = OasisDataset(data_path_training)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_ds = OasisDataset(data_path_testing)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    val_ds = OasisDataset(data_path_validation)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    return train_dl, test_dl, val_dl
