'''Data loader for loading and preprocessing data'''

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

# Load and return normalized data
def load_data(path, img_size, colour):

    dataset = datasets.ImageFolder(root=path, transform=transforms)