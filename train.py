import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from torchvision.transforms import ToTensor
# from torch.autograd import Function
# from itertools import repeat
import numpy as np
# import os
# import pandas as pd
# from torchvision.io import read_image
import modules as m
import dataset as d
import time
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Using cpu.")

# These are the hyper parameters for the training.
epochs = 5
learning_rate = 0.0001
batch = 32

model = m.ModifiedUNet(3, 1).to(device)

img_dir = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2"
seg_dir = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2"
test_dir = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Test_Input"
train_dataset = d.ISICDataset(img_dir, seg_dir, d.transform('train'), d.transform('seg'))
test_dataset = d.ISICDataset(test_dir, seg_dir)
train_loader = DataLoader(train_dataset, batch, shuffle=True)
test_loader = DataLoader(test_dataset, batch)

# We will use the ADAM optimizer
ADAMoptimizer = optim.Adam(model.parameters(), lr=learning_rate)
test_set_dice_list = []
# Now we begin timing
starttime = time.time()
for epoch in range(epochs):
    losslist = []
    runningloss_val = []
    # dicelist = []
    runningloss = 0.0


    model.train()

    for i, input in enumerate(train_loader):
        if i <= 10000:
            image, segment = input[0].to(device), input[1].to(device)

            ADAMoptimizer.zero_grad()

            modelled_image = model(image)[0]
            if i == 1:
                # test_image = image.cpu()[0].numpy()
                test_segment = segment.squeeze().squeeze(0).cpu()
                # test_modelled_image = modelled_image.cpu()
                plt.imshow(test_segment)
                # plt.imshow(test_segment.numpy())
                # plt.imshow(test_modelled_image.numpy())
                plt.savefig("/home/Student/s4742286/PatternAnalysis-2023/outputs/test.jpg")
                exit()
            loss = m.dice_loss(modelled_image, segment)
            loss.backward()
            ADAMoptimizer.step()
            runningloss += loss.item()
            losslist.append(loss.item())
            # if i % 1000 == 0:
            print(f"Training: Epoch {epoch}/{epochs}, Images {i}/10000")
            if i > 300:
                exit()
        elif i in range(10001, 14001):
            if i == 10001:
                print("Validating.")
            with torch.no_grad():
                model.eval()
                images, segments = input[0].to(device), input[1].to(device)

                modelled_images = model(images)[0]
                loss = m.dice_loss(modelled_images, segments)
                runningloss_val.append(loss.item())
            if i % 100 == 0:
                print(f"Validating: Epoch {epoch}/{epochs}, Images {i - 10000}/4000")
        else:
            break

    if epoch in [1, 3, 5]:
        plt.plot(losslist)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss to Epoch {epoch}')
        plt.savefig("/home/Student/s4742286/PatternAnalysis-2023/outputs/Training_Loss_Epoch_{epoch}")
        plt.clf()

        plt.plot(runningloss_val)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Validation Loss to Epoch {epoch}')
        plt.savefig("/home/Student/s4742286/PatternAnalysis-2023/outputs/Validation_Loss_Epoch_{epoch}")


with torch.no_grad():  
    model.eval()
    for i, input in enumerate(train_loader):
        if i < 14000:
            continue
        else:
            images, segments = input[0].to(device), input[1].to(device)
            modelled_images = model(images)[0]
            dice_score = 1 - m.dice_loss(modelled_images, segments)
            test_set_dice_list.append(dice_score.item())



test_dice_score = np.mean(test_set_dice_list)
print(f"Testing finished. Time taken was {time.time() - starttime}. Overall, the dice score that the model was able to provide was {test_dice_score}")  


    

            