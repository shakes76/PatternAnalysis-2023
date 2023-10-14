import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode
from torchvision.io import read_image
import glob
import numpy as np

class GetADNITrain(Dataset):
    def __init__(self, path, train_split=0.9, train=True):
        
        # Get the list of images and sort them
        image_list = sorted(glob.glob(path))

        # Shrink to the desired train_split and if in validation or train
        if train:
            image_list = image_list[:len(image_list) * train_split]
        else:
            image_list = image_list[len(image_list) * train_split:]

        # Get the array length
        self.arrLen = len(image_list)
        self.image_list = image_list

    def __getitem__(self, index):
        # Load the image in grayscale
        x = read_image(self.image_list[index], ImageReadMode.GRAY)

        # Normalize between [0, 1]
        x = x / torch.max(x)

        return x
