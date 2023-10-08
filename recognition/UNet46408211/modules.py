import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import numpy as np
import PIL.Image as Image
import PIL
import os

class DoubleConv(nn.Module):
    """
    A module consisting of two convolutional layers with batch normalization and ReLU activation.
    """
    