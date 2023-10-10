# containing the source code of the components of your model. 
# Each component must be implementated as a class or a function

import torch
import torch.nn as nn

# Build CNN network and get its embedding vector
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # size: 256*240 -> 128*120

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # size: 128*120 -> 64*60

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # size: 64*60 -> 32*30
            )
        
        self.fc = nn.Sequential(
            nn.Linear(128*32*30, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 2),
            )
        
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out


# construct the triplet loss  
class TripletLoss(nn.Module):
    def __ini__(self, margin=1):
        super(TripletLoss, self).__init__()
    
    def forward(self, anchor, positive, negative):
        # calculate euclidean distance from anchor to positive and negative 
        anchor_positive = (anchor - positive).pow(2).sum(1)
        anchor_negative = (anchor - negative).pow(2).sum(1)

        # aclculate loss, use relu to ensure loss are non-negative
        loss = torch.relu(anchor_positive - anchor_negative + self.margin)
        return loss.mean()