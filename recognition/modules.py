import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
# from perceriver_pytorch import Perceriver # install the perceiver

#Hyperparameter
LATENT_DIMENTIONS = 128
EMBEDDED_DIMENTIONS = 32
DEPTH = 4


def createResNet():
    return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)


class CrossAttention(nn.Module):
    def __init__(self):
        super(CrossAttention, self).__init__()

    
    def forward(input_x, input_y):
        v = nn.Linear(input_y)
        k = nn.Linear(input_y)
        return (v, k)
    
class SelfAttention(nn.Module):
    def __init__(self):
        self.attention =nn.MultiheadAttention()

class LatentTransformer(nn.Module):
    def __init__(self):
        super(LatentTransformer, self).__init__()

class PerceiverBlock(nn.Module):
    def __init__(self, depth, latent_dimensions, embedded_dimensions):
        super.__init__()
        self.crossAttention = CrossAttention()
        self.latentTransformerArray = LatentTransformer()
        
        
        self.depth = depth
    
    def forward(self, ):
        pass

class Perceiver(nn.Module):
    def __init__(self, query, key, value):
        self.query = query
        self.key = key
        self.value = value
        self.perceiverBlock = PerceiverBlock(DEPTH, LATENT_DIMENTIONS, EMBEDDED_DIMENTIONS)


class ADNI(nn.Module):
    def __init__(self):
        self.model = [Perceiver for _ in range(DEPTH)]
        self.latent = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.zeros((LATENT_DIMENTIONS, 1, EMBEDDED_DIMENTIONS)),  #embed first then latent dimensions. Others just used torch.empty
                mean=0, 
                std=0.02, 
                a=-2, 
                b=2))
    
    def forward(self, input_x):
        latent = self.latent
        for level in self.model:
            latent = level(latent, )