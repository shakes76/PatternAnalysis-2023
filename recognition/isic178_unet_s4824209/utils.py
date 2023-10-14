'''
source: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
'''

import torch.nn.functional as F
import torch.nn as nn



class Diceloss(nn.Module):
    def __init__(self):
        super(Diceloss, self).__init__()
        self.dice = 0
        
        
    def forward(self, inputs, targets):    
        inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        self.dice = (2.*intersection)/(inputs.sum()+targets.sum())

        return 1 - self.dice
    
