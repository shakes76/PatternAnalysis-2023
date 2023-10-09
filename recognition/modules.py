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
    def __init__(self, dimensions):
        super(CrossAttention, self).__init__()
        self.layer_norm = nn.LayerNorm() # based on some dimension
        self.linear_layer = nn.Linear(dimensions, dimensions) # Pass in input and output neuron amounts
         # takes num_heads which are num of heads, d_model which are dimensions of input and output tensor 
        self.cross_attention = nn.MultiheadAttention(num_heads=1)
    
    def forward(self, latent, key_value):
        # Paper states that inputs are first passed through the layer norm then through linear layers
        result = self.layer_norm(latent)
        result = self.linear_layer(result)
        #cross attention takes 3 parameters: (latent, key, value)
        result = self.cross_attention(result, key_value, key_value)
        return result
    
class SelfAttention(nn.Module):
    def __init__(self, dimensions):
        super(CrossAttention, self).__init__()
        self.layer_norm = nn.LayerNorm() # based on some dimension
        self.linear_layer = nn.Linear(dimensions, dimensions) # Pass in input and output neuron amounts
        # takes num_heads which are num of heads, d_model which are dimensions of input and output tensor 
        # paper states that 8 heads are used per self attention
        self.cross_attention = nn.MultiheadAttention(num_heads=8)
    
    def forward(self, latent):
        # Paper states that inputs are first passed through the layer norm then through linear layers
        result = self.layer_norm(latent)
        result = self.linear_layer(result)
        #cross attention takes 3 parameters: (latent, key, value)
        result = self.cross_attention(result, result)
        return result

class LatentTransformer(nn.Module):
    def __init__(self, latent_depth, dimensions):
        super(LatentTransformer, self).__init__()
        self.depth = latent_depth
        self.self_attention_stack = nn.ModuleList([SelfAttention(dimensions) for i in range(latent_depth)])

    
    def forward(self, latent):
        for self_attention in self.self_attention_stack:
            latent = self_attention.forward(latent)
        return latent

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