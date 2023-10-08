import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np


from modules import SiameseModel, RawSiameseModel, ContrastiveLossFunction
from dataset import load_train_data, load_test_data


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning CUDA not Found. Using CPU")

    ######### LOADING DATA ##########
    print("Begin loading data")
    train_loader = load_train_data()
    test_loader = load_test_data()

    print("Finish loading data")


    #########  TRAINING SIAMASE MODEL ##########
    # Testing model
    model = RawSiameseModel()

    # hyper parameters
    num_epochs = 20
    
    print("Start training")
    for epoch in range(num_epochs):
        print(f"epoch: {epoch + 1}")
        
        for i, val in enumerate(train_loader):
            img0, img1 , label = val
            # data, label = val
            print(model(img0).shape)
            print(model(img1).shape)

            break
        
        break

    print("Finish training")


    #########  TESTING SIAMASE MODEL ##########
    print("Start Testing")

    # evaluate the model
    model.eval() # disable drop out, batch normalization
    with torch.no_grad(): # disable gradient computation
        correct_predict = 0
        total_test = 0
        # for img1, img2, labels in test_loader:
        #     img1 = img1.to(device)
        #     img2 = img2.to(device)
        #     labels = labels.to(device)
        #     outputs = model(img1, img2)
        #     _, predicted = torch.max(outputs.data, 1)
        #     total_test += labels.size(0)
        #     correct_predict += (predicted == labels).sum().item()
        
        # print('Test Accuracy: {} %'.format(100 * correct_predict / total_test))

    print("Finish Testing")

    
    pass