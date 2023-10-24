# Importing necessary libraries and modules
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    """
    A Siamese Neural Network that takes in pair of images and returns vectors 
    for both images in the pair. The vectors are then used to determine the similarity 
    between the two images.

    Args:
        nn (torch.nn.Module)): Base class for all neural network modules in PyTorch.
    """

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=10,stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=1),
            
            nn.Conv2d(32, 64, kernel_size=7, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=1),

            nn.Conv2d(64, 128, kernel_size=4,stride=1),
            nn.ReLU(inplace=True)
        )

        
        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(128*108*100, 64),
            nn.Sigmoid(),
            nn.Linear(64,1),
            nn.Sigmoid()
          
           
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
    
    # Define the Contrastive Loss Function
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function. Computes the contrastive loss between pairs of samples based on their 
    distances and labels.

    Args:
        margin (float): The margin value beyond  which the loss will not incease.
                        It acts as a threshold to separate positive and negative pairs.
                        Default is 2.0.
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):

      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


      return loss_contrastive

