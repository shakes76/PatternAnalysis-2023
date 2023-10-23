'''Source code of the components of your model. Each component must be
implementated as a class or a function
'''

# Importing necessary libraries and modules
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):

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
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidean distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


      return loss_contrastive

