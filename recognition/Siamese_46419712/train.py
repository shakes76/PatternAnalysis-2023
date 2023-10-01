import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np


from modules import SiameseModel
from dataset import load_train_data
from dataset import load_test_data


if __name__ == '__main__':

    train_loader = load_train_data()

    test_loader = load_test_data()
    pass