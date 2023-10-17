import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import math

# from perceriver_pytorch import Perceriver # install the perceiver


class PositionalImageEmbedding(nn.Module):
    def __init__(self, input_shape, input_channels, embed_dim, bands=4):
        super().__init__()
        self.ff = self.fourier_features(
            shape=input_shape, bands=bands)
        self.conv = nn.Conv1d(input_channels + self.ff.shape[0], embed_dim, 1)


    def forward(self, x):
        enc = self.ff.unsqueeze(0).expand(
            (x.shape[0],) + self.ff.shape)
        enc = enc.type_as(x)
        x = torch.cat([x, enc], dim=1)
        x = x.flatten(2)
        x = self.conv(x)
        x = x.permute(2, 0, 1)
        return x
    

    def fourier_features(self, shape, bands):
        dims = len(shape)
        pos = torch.stack(list(torch.meshgrid(
            *(torch.linspace(-1.0, 1.0, steps=n) for n in list(shape))
        )))
        pos = pos.unsqueeze(0).expand((bands,) + pos.shape)
        band_frequencies = (torch.logspace(
            math.log(1.0),
            math.log(shape[0]/2),
            steps=bands,
            base=math.e
        )).view((bands,) + tuple(1 for _ in pos.shape[1:])).expand(pos.shape)
        result = (band_frequencies * math.pi * pos).view((dims * bands,) + shape)
        result = torch.cat([
            torch.sin(result),
            torch.cos(result),
        ], dim=0)

        return result

#Chatgpt generated change later
class GridPositionalImageEmbedding(nn.Module):
    def __init__(self, input_shape, input_channels, embed_dim):
        super(GridPositionalImageEmbedding, self).__init__()
        self.embedding = self.generate_positional_embedding(input_shape, embed_dim)
        self.conv = nn.Conv2d(input_channels + embed_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        enc = self.embedding.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        enc = enc.type_as(x)
        x = torch.cat([x, enc], dim=1)
        x = self.conv(x)
        return x

    def generate_positional_embedding(self, input_shape, embed_dim):
        height, width = input_shape
        num_positions = height * width
        position = torch.arange(num_positions, dtype=torch.float32).reshape(1, 1, height, width).type_as(torch.ones(1))
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim)).type_as(torch.ones(1))
        pos_embedding = torch.cat([torch.sin(position * div_term), torch.cos(position * div_term)], dim=1)
        return pos_embedding

def createResNet():
    return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

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
        # result = result.mean(dim=0)
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
        self.positionalImageEmbedding = PositionalImageEmbedding((240, 240), 1, self.embedded_dimensions)
        
        

    
    def forward(self, kv):
        latent = self.latent.expand(-1, kv.size()[0], -1)
        #kv = self.positionalImageEmbedding(kv)
        kv = kv.view(1800, 5, 32)
        for block in self.perceiver_block_array:
            latent = block(latent, kv)
        #Need to do the classifier here
        latent = self.classifier(latent)
        return latent