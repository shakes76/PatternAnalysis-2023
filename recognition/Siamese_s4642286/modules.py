"""
Name: modules.py
Student: Ethan Pinto (s4642286)
Description: Contains the source code of the components of the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(self).__init__()

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

    def forward(self, input):
        # Forward pass through the Siamese network
        return self.cnn(input)
    

class SiameseNetwork(nn.Module):
    def __init__(self, cnn1, cnn2):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = cnn1
        self.cnn2 = cnn2

    def forward(self, input1, input2):
        output1 = self.cnn1(input1)
        output2 = self.cnn2(input2)
        return output1, output2



class ContrastiveLoss(nn.Module):
  def __init__(self, margin):
    super(ContrastiveLoss, self).__init__()
    self.margin = margin

  def forward(self, y1, y2, label):
    # label = 0 means y1 and y2 are the same
    # label = 1 means y1 and y2 are different
    
    euc_dist = torch.nn.functional.pairwise_distance(y1, y2)

    loss = torch.mean((1-label) * torch.pow(euc_dist, 2) +
      (label) * torch.pow(torch.clamp(self.margin - euc_dist, min=0.0), 2))

    return loss


# Takes in a feature vector, returns either 0 or 1 (AD OR NC)
class MLP(nn.module):
    def __init__(self):
        super(self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, input):
        output1 = nn.ReLU(self.fc1(input))
        output2 = nn.Sigmoid(self.fc2(output1))
        return output2


# Step 3: classification - take in individual images into snn, then write an mlp which takes in a feature vector of 128 and then classifies into one of two classes.
