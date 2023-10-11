# “dataset.py" containing the data loader for loading and preprocessing your data

from typing import Any, Callable, Optional, Tuple
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
import random
from PIL import Image

class DataLoader(object):
    def __init__(self, batch_size=64) -> None:
        self.trainingFile = "AD_NC/train/"
        self.testFile = "AD_NC/test/"
        self.imageSize = 64
        self.batchSize = batch_size
        self.normalisation = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        self.trainloader = None
        self.testloader = None
        
        self.load_training_data()
        self.load_test_data()

    def load_training_data(self):
        # Get training dataset from image folder
        train_images = ImageFolder(root=self.trainingFile, transform=T.Compose([T.Resize(self.imageSize), 
                                    T.CenterCrop(self.imageSize), T.ToTensor(), T.Normalize(*self.normalisation)]),
                                    )
        
        # Get training loader
        self.trainloader = TorchDataLoader(train_images, batch_size=self.batchSize, 
                                 shuffle=True, num_workers=3, pin_memory=True)
        
    def load_test_data(self):
        # Get training dataset from image folder
        test_images = ImageFolder(root=self.testFile, transform=T.Compose([T.Resize(self.imageSize), 
                                    T.CenterCrop(self.imageSize), T.ToTensor(), T.Normalize(*self.normalisation)]),
                                    )

        # Get training loader
        self.testloader = TorchDataLoader(test_images, batch_size=self.batchSize, 
                                 shuffle=True, num_workers=3, pin_memory=True)
        
    def get_training_loader(self):
        return self.trainloader
    
    def get_test_loader(self):
        return self.testloader
    
    def denorm(self, img_tensors):
        return img_tensors * self.normalisation[1][0] + self.normalisation[0][0]
    
    def show_images(self, images, nmax=64):
        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(self.denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))
        plt.show()
        
    def show_batch(self, dl, nmax=64):
        for images, _ in dl:
            self.show_images(images, nmax)
            break
