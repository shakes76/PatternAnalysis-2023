# need to contain data loader and preprocess the data
# i.e. transform the images into tensors and shape them all to the same size

# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image

import time
import math
import matplotlib.pyplot as plt
import os



# create hyper paramaters
new_size = 128
cropCoefficient = 0.9

# create dataset class to store all the images
class ISIC2018DataSet(Dataset):
    def __init__(self):
        




# functions to transform the images
def img_transform():

    img_tr = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize((new_size, new_size))
    ])

    return img_tr

def test_transform():

    test_tr = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize((new_size, new_size))
    ])

    return test_tr

def label_transform():
        
    label_tr = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize((new_size, new_size))
    ])

    return label_tr