"""
Name: modules.py
Student: Ethan Pinto (s4642286)
Description: Contains the source code of the components of the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # CNN and Pooling layers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=10), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=7), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=4), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=4),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten() # Outputs a 128 dimensional feature vector
        )

    def forward(self, x):
        # Forward pass through the Siamese network
        x = self.cnn(x)
        return x


    def other(self,x):

        # Compute L1 Distance between two embeddings
        distance  = torch.abs(output1 - output2)

        # torch pairwise_distance

        # contrastive loss after distance

        # Classify whether the images belong to the same class or different classes
        prediction = self.fc(distance)
        return prediction



class MLP(nn.module):
    def __init__(self):
        super(MLP, self).__init__() 

        self.siamese = SiameseNetwork() # need to load the saved weights of the siamese network here

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        output = self.siamese(input)
        output = self.fc(output)
        return output
    







# write siamese model so that it takes one image and output is feature vector
# but when training, when iterating through trainloader, pass in each image from a pair separately into the SNN.





# MLP SHOULD BE TRAINED ON THE FEATURE VECTOR.