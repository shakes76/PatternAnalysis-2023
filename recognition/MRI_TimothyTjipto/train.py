'''Source code for training, validating, testing and saving your model. The model
should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
sure to plot the losses and metrics during training'''

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random

import dataset
from modules import SiameseNetworkDataset, SiameseNetwork, ContrastiveLoss
from predict import show_plot
import matplotlib.pyplot as plt
import numpy as np
import random


import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# Paths of data 
# TRAIN_AD_PATH = "C:/Users/barkz/Downloads/ADNI_AD_NC_2D/AD_NC/train/AD" 
# TRAIN_NC_PATH =  "C:/Users/barkz/Downloads/ADNI_AD_NC_2D/AD_NC/train/ND" 

TRAIN_PATH = "C:/Users/barkz/Downloads/ADNI_AD_NC_2D/AD_NC/train"

TEST_AD_PATH =  "C:/Users/barkz/Downloads/ADNI_AD_NC_2D/AD_NC/test/AD" 
TEST_NC_PATH = "C:/Users/barkz/Downloads/ADNI_AD_NC_2D/AD_NC/test/ND" 

INPUT_SHAPE= (120, 128) # SIZE OF IMAGE 256 X 240
COLOR_MODE= 'grayscale'

BATCH_SIZE = 16
TRAINING_SIZE= 1800

TRAINING_MODE = True
VISUALISE = True

# def main():
   
#     # AD_dataset = dataset.normalise_data(TRAIN_AD_PATH,INPUT_SHAPE)
#     # NC_dataset = dataset.normalise_data(TRAIN_NC_PATH,INPUT_SHAPE)
    
#     TRAIN_DATASET = dataset.normalise_data(TRAIN_PATH, INPUT_SHAPE)
#     Same_class = random.randint(0,1)

#     pos_pair1, pos_pair2, neg_pair1, neg_pair2 =dataset.make_pair(AD_dataset, NC_dataset)
#     mixed_dataset = dataset.shuffle(pos_pair1, pos_pair2, neg_pair1, neg_pair2)
#     train_dataset,validation_dataset = dataset.split_dataset(mixed_dataset,TRAINING_SIZE)

#     train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
#     validation_dataloader = DataLoader(validation_dataset,batch_size=BATCH_SIZE,shuffle=True)

#     train_features, train_labels = next(iter(train_dataloader))
    
#     print(f"Feature batch shape: {train_features.size()}")
#     print(f"Labels batch shape: {train_labels.size()}")
#     img = train_features[0].squeeze()
#     label = train_labels[0]
#     plt.imshow(img, cmap="gray")
#     plt.show()
#     print(f"Label: {label}")

def main():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale with one channel
        transforms.Resize((120,128)),  # img_size should be a tuple like (128, 128) actual img(256x240)
        transforms.ToTensor(),
        # You can add more transformations if needed
    ])
    siamese_dataset = SiameseNetworkDataset(TRAIN_PATH, transform)
    # Create a simple dataloader just for simple visualization
    vis_dataloader = DataLoader(siamese_dataset,
                            shuffle=True,
                            num_workers=2,
                            batch_size=8)

    train_dataloader = dataset.load_data(siamese_dataset,BATCH_SIZE,8,True)
    net = SiameseNetwork().cuda()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.0005)


    counter = []
    loss_history = [] 
    iteration_number= 0

    # Iterate throught the epochs
    for epoch in range(100):

        # Iterate over batches
        for i, (img0, img1, label) in enumerate(train_dataloader, 0):

            # Send the images and labels to CUDA
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

            # Zero the gradients
            optimizer.zero_grad()

            # Pass in the two images into the network and obtain two outputs
            output1, output2 = net(img0, img1)

            # Pass the outputs of the networks and label into the loss function
            loss_contrastive = criterion(output1, output2, label)

            # Calculate the backpropagation
            loss_contrastive.backward()

            # Optimize
            optimizer.step()

            # Every 10 batches print out the loss
            if i % 10 == 0 :
                print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
                iteration_number += 10

                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())

        show_plot(counter, loss_history)

if __name__ == '__main__':
    main()