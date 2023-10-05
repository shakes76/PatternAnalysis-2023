# This is the train file

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import train_loader, test_loader
from modules import VisionTransformer
import matplotlib.pyplot as plt

# Checking for GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")