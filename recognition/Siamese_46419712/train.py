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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning CUDA not Found. Using CPU")

    ######### LOADING DATA ##########
    print("Begin loading data")
    train_loader = load_train_data()
    test_loader = load_test_data()

    print("Finish loading data")

    num_epochs = 20

    #########  TRAINING SIAMASE MODEL ##########
    # Testing model
    model = SiameseModel()
    
    print("Start training")
    for epoch in range(num_epochs):
        print(f"epoch: {epoch + 1}")
        
        for i, val in enumerate(train_loader):
            # img0, img1 , label = data
            data, label = val
            model(data)

            break
        
        break

    print("Finish training")


    #########  TESTING SIAMASE MODEL ##########
    print("Start Testing")

    print("Finish Testing")

    
    pass