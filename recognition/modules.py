import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import math


class MultilayerPerceptron(nn.Module):
    def __init__(self, dimensions):
        super(MultilayerPerceptron, self).__init__()
        self.layer_normalisation = nn.LayerNorm(dimensions) # First step in paper for MLP is the layer normalisation
        self.linear_layer1 = nn.Linear(dimensions, dimensions) # Then inputs are passed through 2 linear layers
        self.linear_layer2 = nn.Linear(dimensions, dimensions)
        self.gelu_act = nn.GELU() # Paper recommendeds using  GELU activation function
        #Optional dropout function can go here
    
    def forward(self, input):
        result = self.layer_normalisation.forward(input)
        result = self.linear_layer1(result)
        result = self.gelu_act(result)
        result = self.linear_layer2(result)
        return result


class CrossAttention(nn.Module):
    """
        This class performs the cross attention function of the model which is the first block in the diagram of the achitecture.
        This consists of passing latent array through the layer norm and then performing the cross attention on latent and key_value

    """
    def __init__(self, embedded_dimensions, cross_attention_heads):
        super(CrossAttention, self).__init__()
        self.layer_norm = nn.LayerNorm(embedded_dimensions) # Paper states the cross attention inputs first go through layer norm
         # takes num_heads which are num of heads, d_model which are dimensions of input and output tensor 
        self.cross_attention = nn.MultiheadAttention(embed_dim=embedded_dimensions, num_heads=cross_attention_heads) # Paper states this runs with single head
        self.multilayerPerceptron = MultilayerPerceptron(embedded_dimensions) # Paper states the the inputs are passed through a dense layer next
    
    def forward(self, latent, key_value):
        """
        This performs the actions of the multihead attention on the latent and key_value.
        This learns features of both arrays.
        """
        # Paper states that inputs are first passed through the layer norm then through linear layers
        result = self.layer_norm(latent)
        #cross attention takes 3 parameters: (latent, key, value)
        result = self.cross_attention(result, key_value, key_value)[0]
        result = self.multilayerPerceptron(result)
        return result
    
class SelfAttention(nn.Module):
    """
        This class performs the self attention aspect of the model. This is used in the latent transformer step of the architecture.
        This learns features and only takes the latent array as parameter for the forward method.
    """
    def __init__(self, embedded_dimension, self_attention_heads):
        super(SelfAttention, self).__init__()
        self.layer_norm = nn.LayerNorm(embedded_dimension) # based on some dimension
        # takes num_heads which are num of heads, d_model which are dimensions of input and output tensor 
        # paper states that 8 heads are used per self attention
        self.multilayerPerceptron = MultilayerPerceptron(embedded_dimension)
        self.cross_attention = nn.MultiheadAttention(embed_dim=embedded_dimension, num_heads=self_attention_heads)
    
    def forward(self, latent):
        """
            This only takes the latent and learns specific features about itself.
            This self attention is run multiheaded with 4 heads based on hyperparameters in train.py
        """
        # Paper states that inputs are first passed through the layer norm then through linear layers
        result = self.layer_norm(latent)
        #cross attention takes 3 parameters: (latent, key, value)
        result = self.cross_attention(result, result, result)[0]
        result = self.multilayerPerceptron(result)
        return result

class LatentTransformer(nn.Module):
    """
        This is a series of self attentions performed in a array. This extracts features and reduces size of the output afterwards. 
    """
    def __init__(self, self_attention_depth, self_attention_heads, embedded_dimensions):
        super(LatentTransformer, self).__init__()
        self.self_attention_stack = nn.ModuleList([SelfAttention(embedded_dimensions, self_attention_heads) for i in range(self_attention_depth)])

    
    def forward(self, latent):
        """
            This function only takes the latent array for the multiheaded self attention
        """
        for self_attention in self.self_attention_stack:
            latent = self_attention.forward(latent)
        return latent

class Block(nn.Module):
    """
        This class contains the block of main functionality of the model. It implements the cross attention and latent transformer steps.
        This performs all the cross attention and self attention steps. 
    """
    def __init__(self, self_attention_depth, latent_dimensions, embedded_dimensions, cross_attention_heads, self_attention_heads):
        super(Block, self).__init__()
        self.crossAttention = CrossAttention(embedded_dimensions, cross_attention_heads)
        self.latentTransformerArray = LatentTransformer(self_attention_depth, self_attention_heads, embedded_dimensions)
    
    def forward(self, latent, key_value):
        """
            The key_value and latent are given to the cross_attention to learn features about both then the latent learns further features
            from itself in latenttransformer.
        """
        result = self.crossAttention(latent, key_value)
        result = self.latentTransformerArray(result)
        return result


class Classifier(nn.Module):
    """
        This class is final section to determine the whether the image contains Alzheimer features.
        Passes the latent array through 2 linear layers and returns a binary output.
    """
    def __init__(self, embedded_dimensions, n_classes=2):
        super(Classifier, self).__init__()
        self.linear_layer1 = nn.Linear(embedded_dimensions, embedded_dimensions)
        self.linear_layer2 = nn.Linear(embedded_dimensions, n_classes)

    def forward(self, latent):
        """
            The mean function here aggregates the results from first layer to return the binary output for the second linear layer
        """
        result = self.linear_layer1(latent)
        result = result.mean(dim=0) #Needed to reduce tensor to 1d from 2d
        return self.linear_layer2(result)

class Perceiver(nn.Module):
    """
        This class pieces all the other classes together.
        This class is initialised in the train.py file and used for training.
        All the hyperparameters passed in here determine the running of the other classes in the file.
        This constructs all other classes required for the functionality of the model.
    """
    def __init__(self, model_depth, self_attention_depth, latent_dimensions, embedded_dimensions, cross_attention_heads, self_attention_heads):
        super(Perceiver, self).__init__()
        self.model_depth = model_depth
        self.self_attention_depth = self_attention_depth
        self.latent_dimensions = latent_dimensions
        self.embedded_dimensions = embedded_dimensions
        self.cross_attention_heads = cross_attention_heads
        self.self_attention_heads = self_attention_heads
        self.classifier = Classifier(embedded_dimensions)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.perceiver_block_array = nn.ModuleList([Block(self.self_attention_depth, self.latent_dimensions, self.embedded_dimensions, self.cross_attention_heads, self.self_attention_heads) for i in range(self.model_depth)])
        self.latent = torch.zeros((self.latent_dimensions, 1, self.embedded_dimensions)).to(self.device)
        
    def forward(self, kv):
        """
            This expands the latent array to match size of key_value. Then the key_value is reshaped for the cross attention within the block of 
            perceiver block array. Finally the resulting latent array is passed through the classifier and returneds the output.
        """
        latent = self.latent.expand(-1, kv.size()[0], -1)
        kv = kv.view(1800, 5, 32) # Restructures the kv input to have batch size and embedded dimensions
        for block in self.perceiver_block_array:
            latent = block(latent, kv)
        latent = self.classifier(latent)
        return latent