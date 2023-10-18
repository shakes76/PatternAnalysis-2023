#import modules
import argparse
import os
import random
import numpy as np

import dataset
import modules
import train
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from glob import glob

from PIL import Image


# Device



def main():
    CUDA_DEVICE_NUM = 0
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device:', DEVICE)
    print(torch.cuda.is_available())

    # Hyperparameters
    LEARNING_RATE = 0.0001
    TRAINDATA = "ISIC/ISIC-2017_Training_Data/ISIC-2017_Training_Data"
    TESTDATA = "ISIC/ISIC-2017_Test_v2_Data/ISIC-2017_Test_v2_Data"
    VALIDDATA = "ISIC/ISIC-2017_Validation_Data/ISIC-2017_Validation_Data"
    TRAINTRUTH = "ISIC/ISIC-2017_Training_Part1_GroundTruth/ISIC-2017_Training_Part1_GroundTruth"
    TESTTRUTH = "ISIC/ISIC-2017_Test_v2_Part1_GroundTruth/ISIC-2017_Test_v2_Part1_GroundTruth"
    VALIDTRUTH = "ISIC/ISIC-2017_Validation_Part1_GroundTruth/ISIC-2017_Validation_Part1_GroundTruth"

    NUM_EPOCHS = 5
    BATCH_SIZE = 4
    WORKERS = 4

    train_dataset = dataset.CustomDataset(image_dir = TRAINDATA,
                                    mask_dir=TRAINTRUTH,
                                    transform=transforms.Compose([                                    
                                    transforms.ToTensor()]))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                            shuffle=True)

    valid_dataset = dataset.CustomDataset(image_dir = VALIDDATA,
                                    mask_dir=VALIDTRUTH,
                                    transform=transforms.Compose([                                    
                                    transforms.ToTensor()]))


    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE,
                                            shuffle=True)
    '''
    test_dataset = dataset.CustomDataset(image_dir = TESTDATA,
                                    mask_dir=TESTTRUTH,
                                    transform=transforms.Compose([                                    
                                    transforms.ToTensor()]))


    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                            shuffle=True)
    '''
    FILE = "model.pth"
    loaded_model = train.UNet(3,1,[64,128,256,512]) 
    loaded_model.load_state_dict(torch.load(FILE))
    loaded_model.to(DEVICE)
    loaded_model.eval()
    utils.save_predictions_as_imgs(valid_dataloader,loaded_model)
    utils.check_accuracy(valid_dataloader,loaded_model,folder="saved_images/")



if __name__ == "__main__":
    main()