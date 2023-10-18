import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import math

# from perceriver_pytorch import Perceriver # install the perceiver

class MultilayerPerceptron(nn.Module):
    def __init__(self, dimensions):
        super(MultilayerPerceptron, self).__init__()
        self.layer_normalisation = nn.LayerNorm(dimensions) # Needs some parameters
        self.linear_layer1 = nn.Linear(dimensions, dimensions) # Pass in input and output neuron amounts
        self.linear_layer2 = nn.Linear(dimensions, dimensions)
        self.gelu_act = nn.GELU()
        #Optional dropout function can go here
    
    def forward(self, input):
        result = self.layer_normalisation.forward(input)
        result = self.linear_layer1(result)
        result = self.gelu_act(result)
        result = self.linear_layer2(result)
        return result


class CrossAttention(nn.Module):
    def __init__(self, embedded_dimensions, cross_attention_heads):
        super(CrossAttention, self).__init__()
        self.layer_norm = nn.LayerNorm(embedded_dimensions) # based on some dimension
         # takes num_heads which are num of heads, d_model which are dimensions of input and output tensor 
        self.cross_attention = nn.MultiheadAttention(embed_dim=embedded_dimensions, num_heads=cross_attention_heads)
        self.multilayerPerceptron = MultilayerPerceptron(embedded_dimensions) # needs some parameters
    
    def forward(self, latent, key_value):
        # Paper states that inputs are first passed through the layer norm then through linear layers
        result = self.layer_norm(latent)
        #cross attention takes 3 parameters: (latent, key, value)
        result = self.cross_attention(result, key_value, key_value)[0]
        result = self.multilayerPerceptron(result)
        return result
    
class SelfAttention(nn.Module):
    def __init__(self, embedded_dimension, self_attention_heads):
        super(SelfAttention, self).__init__()
        self.layer_norm = nn.LayerNorm(embedded_dimension) # based on some dimension
        # takes num_heads which are num of heads, d_model which are dimensions of input and output tensor 
        # paper states that 8 heads are used per self attention
        self.multilayerPerceptron = MultilayerPerceptron(embedded_dimension)
        self.cross_attention = nn.MultiheadAttention(embed_dim=embedded_dimension, num_heads=self_attention_heads)
    
    def forward(self, latent):
        # Paper states that inputs are first passed through the layer norm then through linear layers
        result = self.layer_norm(latent)
        #cross attention takes 3 parameters: (latent, key, value)
        result = self.cross_attention(result, result, result)[0]
        result = self.multilayerPerceptron(result)
        return result

class LatentTransformer(nn.Module):
    def __init__(self, self_attention_depth, self_attention_heads, embedded_dimensions):
        super(LatentTransformer, self).__init__()
        self.self_attention_stack = nn.ModuleList([SelfAttention(embedded_dimensions, self_attention_heads) for i in range(self_attention_depth)])

    
    def forward(self, latent):
        for self_attention in self.self_attention_stack:
            latent = self_attention.forward(latent)
        return latent

class Block(nn.Module):
    def __init__(self, self_attention_depth, latent_dimensions, embedded_dimensions, cross_attention_heads, self_attention_heads):
        super(Block, self).__init__()
        self.crossAttention = CrossAttention(embedded_dimensions, cross_attention_heads)
        self.latentTransformerArray = LatentTransformer(self_attention_depth, self_attention_heads, embedded_dimensions)
    
    def forward(self, latent, key_value):
        result = self.crossAttention(latent, key_value)
        result = self.latentTransformerArray(result)
        return result


class Classifier(nn.Module):
    def __init__(self, embedded_dimensions, n_classes=2):
        super(Classifier, self).__init__()
        self.linear_layer1 = nn.Linear(embedded_dimensions, embedded_dimensions)
        self.linear_layer2 = nn.Linear(embedded_dimensions, n_classes)

    def forward(self, latent):
        result = self.linear_layer1(latent)
        result = result.mean(dim=0) #Needed to reduce tensor to 1d from 2d
        return self.linear_layer2(result)

class Perceiver(nn.Module):
    def __init__(self, model_depth, self_attention_depth, latent_dimensions, embedded_dimensions, cross_attention_heads, self_attention_heads):
        super(Perceiver, self).__init__()
        self.model_depth = model_depth
        self.self_attention_depth = self_attention_depth
        self.latent_dimensions = latent_dimensions
        self.embedded_dimensions = embedded_dimensions
        self.cross_attention_heads = cross_attention_heads
        self.self_attention_heads = self_attention_heads
        self.classifier = Classifier(embedded_dimensions)
        self.perceiver_block_array = nn.ModuleList([Block(self.self_attention_depth, self.latent_dimensions, self.embedded_dimensions, self.cross_attention_heads, self.self_attention_heads) for i in range(self.model_depth)])
        #embed first then latent dimensions. Others just used torch.empty
        self.latent = nn.Parameter(
            torch.nn.init.trunc_normal_(torch.zeros((self.latent_dimensions, 1, self.embedded_dimensions)), mean=0, std=0.02, a=-2, b=2)
        )
        
    def forward(self, kv):
        latent = self.latent.expand(-1, kv.size()[0], -1)
        kv = kv.view(1800, 5, 32) # Restructures the kv input to have batch size and embedded dimensions
        for block in self.perceiver_block_array:
            latent = block(latent, kv)
        latent = self.classifier(latent)
        return latent