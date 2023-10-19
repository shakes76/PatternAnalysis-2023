# source code for training, validating, testing and saving the model
import dataset
import modules
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np

# Hyper-parameters
num_epochs = 30
learning_rate = 5 * 10**-4
batchSize = 16
learning_rate_decay = 0.985


# set up the funcitonality from the imported dataset.py and modules.py

validationImagesPath = "isic_data/ISIC2018_Task1-2_Validation_Input"
trainImagesPath = "isic_data/ISIC2018_Task1-2_Training_Input_x2"
validationLabelsPath = "isic_data/ISIC2018_Task1_Validation_GroundTruth"
trainLabelsPath = "isic_data/ISIC2018_Task1_Training_GroundTruth_x2"

def init():
    validDataSet = dataset.ISIC2018DataSet(validationImagesPath, validationLabelsPath, dataset.img_transform(), dataset.label_transform())
    validDataloader = DataLoader(validDataSet, batch_size=batchSize, shuffle=False)
    trainDataSet = dataset.ISIC2018DataSet(trainImagesPath, trainLabelsPath, dataset.img_transform(), dataset.label_transform())
    trainDataloader = DataLoader(trainDataSet, batch_size=batchSize, shuffle=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning CUDA not Found. Using CPU")

    dataLoaders = dict()
    dataLoaders["valid"] = validDataloader
    dataLoaders["train"] = trainDataloader

    dataSets = dict()
    dataSets["valid"] = validDataSet
    dataSets["train"] = trainDataSet

    return dataSets, dataLoaders, device

def main():
    dataSets, dataLoaders, device = init()
    model = modules.Improved2DUnet()
    model = model.to(device)

# training


# validating


# testing


# saving




if __name__ == "__main__":
    main()