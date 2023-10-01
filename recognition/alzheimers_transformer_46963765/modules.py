import torch.nn as nn
import torch.nn.functional as F
import dataset as ds
import torch
import torchvision.models as models


#Resnet Class (50 maybe or 25)
class ADNI_Transformer(nn.Module):
    
    def __init__(self):
        super().__init__()    
        # don't want it pretrianed        
        network = models.resnet34(pretrained=False) 
        # take out the classification layer 
        self._resnet = torch.nn.Sequential(*list(network.children())[:-1])

    def forward(self, images):
        # shape 32x3x240x240
        output = self._resnet(images)
        # reshapes to 32x512x1x1
        
        # use perceiver transformer  
        
        #sigmoid to classify true / false      
        output = torch.sigmoid(output)
        return output



