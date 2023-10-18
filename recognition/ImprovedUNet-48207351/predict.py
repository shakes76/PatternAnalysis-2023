import torch
import torch.nn.functional as F  
import matplotlib.pyplot as plt
from dataset import ISICDataLoader
from modules import ImprovedUNET
import numpy as np

def dice_coefficient(predicted, target, smooth=1.0):
    intersection = np.sum(predicted * target)
    union = np.sum(predicted) + np.sum(target)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

