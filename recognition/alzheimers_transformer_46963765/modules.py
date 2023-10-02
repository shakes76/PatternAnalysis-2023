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


class CrossAttention(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
        
        
    def forward(self, latent, image):
        pass

class LatentTransformer(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
        
        
    def forward(self, latent, image):
        pass
    
    
class Classifier(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
        
        
    def forward(self, latent, image):
        pass



class Perceiver(nn.Module):
    
    
    
    def __init__(self, latent_dim, embed_dim, depth) -> None:
        super().__init__()
        
        LATENT_DIM = 128
        LATENT_EMB = 64
        
        self.latent = torch.empty(LATENT_DIM, LATENT_EMB)
        self._depth = depth
        
        self._crossAttention = CrossAttention()
        self._latentTransformer = LatentTransformer()
        self._classifier = Classifier()
        

        
        
        
    def forward(self, latent, image):
        pass


