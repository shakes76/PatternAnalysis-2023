"""
Name: modules.py
Student: Ethan Pinto (s4642286)
Description: Contains the source code of the components of the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#create the Siamese Neural Network
class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11,stride=4),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3,stride=1),
       )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(3456, 1024),
            nn.ReLU(),
            
            nn.Linear(1024, 256),
            nn.ReLU(),
            
            nn.Linear(256,128)
        )
        
    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

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
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

