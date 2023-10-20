import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision.datasets as datasets
import torch.utils.data as util_data
from torch.utils.data import DataLoader, Dataset
import torch
import random
from PIL import Image
import module as md
import train as tr

if __name__ == '__main__':
    #device configuration
    device = torch.device("cuda")

    #Define parameters for model
    layer = "VGG16"
    in_channels = 1
    classes = 2
    epochs = 5
    learning_rate = 1e-5  

    model = md.Siamese(layers=layer, in_channels=in_channels, classes=classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    tr.model_train(model, optimizer, epochs)
    tr.model_test(model)

