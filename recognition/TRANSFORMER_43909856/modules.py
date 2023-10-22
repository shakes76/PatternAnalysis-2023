import os
import os.path as osp
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


"""
This file contains all of the components required to create a 2D image recognition
transformer (ViT) used for a binary classification problem.
ViT for ImageNet: https://arxiv.org/abs/2205.01580

- ViT has a set of tokens + 1 class token. In this model, average pooling of the
model will be used instead of a class token
- Uses an average pooling layer at the end of all convolution layers. 
Generates a 1 dimensional string used for outputting classification of images
- Image transformers don't use a masked multi-head attention component,
as they don't need to be auto-regressive (look at both information from both the
past and the future of the current position in the input). This transformer is
bi-directional


Possible hyperparams:
- How big the MLP depth is
- No. of attention heads
- Width of the network
- How many attention layers

This model is using S/16 configuration from the ViT paper above


Einops:
- Use to perform dot products on particular indices of passed tensors
using Einstein summation
- Use to transform from 1D to 2D, patchifying, etc.


A rough diagram of required components:

Main sub-component #1: 
->
-> Multi-Head Attention -> Add & Layer Norm -> Feed Forward --> Add & Layer Norm ->
->                               ^                           |         ^  
---------------------------------|                           |---------|


Main sub-component #2: 
->
-> Masked Multi-Head Attention -> Add & Layer Norm -> Multi-Head Attention ------> Add & Layer Norm -> Feed Forward --> Add & Layer Norm ->
->                                      ^                                  ^ ^ |         ^                          |         ^  
----------------------------------------|                                  | | |---------|                          |---------|
                                                                           |-|
                                                                            |

Full ViT:
Inputs -> Input Embedding -> Positional Encoding -> Main sub-component #1 ------
Outputs (shifted right) -> Output Embedding -> Positional Encoding -> Main sub-component #2 -> Linear -> Softmax -> Output probabilities
"""


"""
Creates the multi-head attention modules used within the ViT network.
The component of the network taking inputs will need N multi-head attention modules.
The component of the network taking outputs (also connected to the previous component)
will need N masked multi-head attention modules, and N multi-head attention modules.

Also includes the components for calculating the scaled dot product attention
from the input Keys, Queries, and Values.

Scaled Dot-Product Attention:
Q ->
    MatMul -> Scale -> Mask (optional) -> SoftMax -> Matmul ->
K ->                                                  ^
V ----------------------------------------------------|

Multi-Head Attention:
V -> Linear ->
K -> Linear -> Scaled Dot-Product Attention -> Concat -> Linear ->
Q -> Linear ->

A masked multi-head attention could be created by adding a mask layer to the 
scaled dot product attention component, but a mask layer will not be used for
this model.
"""
class Attention(nn.Module):

    """
    Create/initialise a multi-head attention module, using self-attention.

    Params:
        dimensions (int): dimensions/size of the input data
        n_heads (int): the number of heads added to each multi-head attention component
        head_dimensions (int): the dimensions/size of each head added to the attention.    
    """
    def __init__(self, dimensions, n_heads=8, head_dimensions=64):
        super().__init__()

        self.n_heads = n_heads
        # Normalise the matrix, using the square root of the size of the head
        self.scale = head_dimensions ** (-0.5)
        # All operations will be normalised (layer norm for 1D representations, similar to batch norm)
        self.layer_norm = nn.LayerNorm(dimensions)
        # Used for performing network concatenations
        inner_dimensions = head_dimensions * n_heads
        
        # Softmax layer for each scaled dot product attention (applied before matmul out)
        self.attend = nn.Softmax(dim=-1)
        # Converts every token to a Query, Key, or Value
        self.to_qkv = nn.Linear(dimensions, inner_dimensions * 3, bias=False)
        # After concatenating the scaled dot product attention, concatenate this into a linear layer
        self.to_out = nn.Linear(inner_dimensions, dimensions, bias=False)


    """
    Perform one forward pass step (forward propagation) to train the attention
    module. Create Keys, Queries, and Values from the input data.

    Params:
        x: 1D representation of input data (usually in the form of tokens)
    Returns:
        Computed result of attention module (after Linear flattening layer is applied)
    """
    def forward(self, x):
        # Normalise the input data
        x = self.layer_norm(x)
        '''
        Convert the input into Keys, Queries, and Values.
        If using cross-attention, QKV would be split between x and y, where y
        is another set of tokens.
        '''
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        '''
        Convert the KQV tensors into groups, then split them 
        across the 8 attention heads.
        b - dimensions/size of each batch
        n - TODO
        h - number of heads
        d - dimensions/size of each head
        '''
        q, k, v = map(lambda i: rearrange(i, "b n (h d) -> b h n d", h=self.n_heads), qkv)

        # Get correlations - perform matrix multiplication between Q and K. Scale result to [0, 1]
        q_k_correlations = torch.matmul(q, k.transpose(1, -2)) * self.scale
        # Turn correlations into probabilites using softmax function
        attention = self.attend(q_k_correlations)
        # Multiply attention probabilites with the Values
        out = torch.matmul(attention, v)
        # Concatenate results with # of heads and the dimensions of each head
        out = rearrange(out, "b h n d -> b n (h d)")
        # Apply to Linear layer to give flattened linear output
        return self.to_out(out)


"""
A simple feed-forward NN, used within components of the ViT.

Network contains two hidden Linear layers, with an activation function
between them. Layer normalisation (for 1D data) is also applied to input values.

FeedForward modules are added to the network after multi-head attention modules,
taking the flattened Linear layer outputs from these modules.
One FeedForward module will be placed after the multi-head attention module handling
inputs. A second module will be placed after the chained masked multi-head attention
and multi-head attention modules handling outputs.
As there are N complete network components handling inputs and N complete network
components handling outputs, this means that N FeedForward modules are required
for inputs and N FeedForward modules are required for outputs.
"""
class FeedForward(nn.Module):

    """
    Create the simple linear layers used within the ViT.

    Params:
        dimensions (int): the size/dimensions of the input data
        hidden_layer_dimensions (int): the size/dimensions of the two hidden 
                                       Linear layers
    """
    def __init__(self, dimensions, hidden_layer_dimensions):
        super().__init__()
        # Create the network:
        self.network = nn.Sequential(
            # Apply 1D layer normalisation (similar to batch norm for 2D data)
            nn.LayerNorm(dimensions),
            # Add the first hidden Linear layer
            nn.Linear(dimensions, hidden_layer_dimensions),
            # Apply GELU (Gaussian Error Linear Unit) activation fn
            # TODO why GELU? Try ReLU or other activation fns
            nn.GELU(),
            # Add second hidden Linear layer
            nn.Linear(hidden_layer_dimensions, dimensions)
        )
    
    """
    Perform one forward pass (forward propagation) on the Feed-Forward NN.

    Params:
        x: 1D representation of input data
    Returns:
        Computed result of attention module (after second Linear layer is applied)
    """
    def forward(self, x):
        return self.network(x)


"""
Create the whole Transformer (ViT) network, using combinations of the Attention
and FeedForward modules.
"""
class Transformer(nn.Module):

    """
    Create the layers required for the Transformer network.
    Add Attention modules, whose outputs are fed into FeedForward modules.

    Params:
        dimensions (int): the size/dimensions of the input data
        depth (int): the depth of the network (number of required Attention modules,
                     whose outputs are chained into FeedForward modules)
        n_heads (int): the number of heads added to each multi-head attention component
        head_dimensions (int): the dimensions/size of each head added to the attention. 
    """
    def __init__(self, dimensions, depth, n_heads, head_dimensions, mlp_dimensions):
        super().__init__()
        # All operations will be normalised (layer norm for 1D representations, similar to batch norm)
        self.layer_norm = nn.LayerNorm(dimensions)

        # Add the # of required chained Attention and FeedForward modules
        self.layers = nn.ModuleList([])
        for i in range (depth):
            self.layers.append(nn.ModuleList([
                Attention(dimensions=dimensions, n_heads=n_heads, 
                          head_dimensions=head_dimensions),
                FeedForward(dimensions=dimensions, hidden_layer_dimensions=mlp_dimensions)
            ]))

    
    """
    Perform one forward pass (forward propagation) on the Transformer network.
    Residual connections are maintained between the input to that sub-component
    and the current Attention and FeedForward layers - the residual connection is
    added to the output, then normalised.

    Params:
        x: 1D representation of input data
    Returns:
        Computed result of final Attention module (after second Linear layer of
        final FeedForward module is applied)
    """
    def forward(self, x):
        for attention, feed_forward in self.layers:
            # Add residual connections between the input to that sub-component and the modules
            x = attention(x) + x
            x = feed_forward(x) + x
        # Normalise the output
        return self.layer_norm(x)


"""
Creates a positional encoding for the Transformer input data, using a 2D set of 
sinusoids.
Every row of the encoding will vary with frequency, allowing for the inputs to
be encoded uniquely and their position located.

Params:
    height (int): the required height of the positional encoding
    width (int): the required width of the positional encoding
    dimensions (int): the dimensions/size of the input features. This value 
                      must be a multiple of 4.
    temperature (int): determines the frequencies used by the sinusoids in the
                       positional encoding
    dtype (torch dtype): the data type for the positional encoding to be stored as

Returns:
    The computed positional encoding
"""
def create_positonal_encodings(height, width, dimensions, temperature=10000,
                                dtype=torch.float32):
    # Set up a 2D set of coordinates in a mesh grid
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    # Set the frequencies used by the sinusoids in the positional encoding
    omega = torch.arange(dimensions // 4) / (dimensions // 4 - 1)
    omega = 1.0 / (temperature ** omega)
    
    # Flatten the x and y coordinates into 1D arrays
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    
    # Compute sinusoids and combine them together
    positional_encoding = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return positional_encoding.type(dtype)


"""
Create a simple ViT as mentioned in this paper: https://arxiv.org/abs/2205.01580
The created model will be used in a classification problem.
Using model S/16 (width=384, depth=12, mlp_head_size=1536, n_heads=6)
"""
class SimpleViT(nn.Module):
    """
    Initialise/create a simple ViT model.
    Breaks each image up into smaller sized 'patches', which are used as input
    tokens.

    Params:
        image_size (tuple(int, int)): the size/dimensions of the input image
                                      (height x width)
        patch_size (tuple(int, int)): the size/dimensions of the image patches
                                      (height x width). The image height should 
                                      be a multiple of the patch height, and
                                      the image width should be a multiple of 
                                      the patch width.
        n_classes (int): the number of classes in the classification problem
        dimensions (int):
        depth (int):
        n_heads (int):
        mlp_dimensions (int):
        n_channels (int): the number of channels in the input image (3 for RGB)
        head_dimensions (int):
    """
    def __init__(self, *, image_size, patch_size, n_classes, dimensions, depth,
                    n_heads, mlp_dimensions, n_channels, head_dimensions):
        super().__init__()
        '''
        The image height should be a multiple of the patch height, and
        the image width should be a multiple of the patch width.
        '''
        image_height, image_width = image_size
        patch_height, patch_width = patch_size

        # Get the dimensions of each patch
        patch_dimensions = n_channels * patch_height * patch_width
        
        '''
        Turn all images into multiple patches ('patchifying'), of the size 
        (patch height x patch width x num channels). THe patches are 1D tokens.
        h - height of image
        w - width of image
        p1 - patch height
        p2 - patch width
        '''
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> (p1 p2 c)", p1=patch_height, p2=patch_width),
            # Add a layer norm for 1D dat
            nn.LayerNorm(patch_dimensions),
            # Embed patches in linear layer
            nn.Linear(patch_dimensions, dimensions),
            # Add layer norm after the linear layer
            nn.LayerNorm(dimensions),
        )

        # Create the positional embedding (scale image dimensions by the patch size)
        self.positional_embedding = create_positonal_encodings(
            height=(image_height // patch_height),
            width=(image_width // patch_width),
            dimensions=dimensions
        )

        # Add the Transformer network
        self.transformer = Transformer(dimensions=dimensions, depth=depth, n_heads=n_heads,
                                       head_dimensions=head_dimensions, 
                                       mlp_dimensions=mlp_dimensions)
        
        # Use average pooling for the network (instead of using a class token)
        self.pooling = "mean"

        # Store the identity to perform skip connections
        self.to_latent = nn.Identity()

        # Linear layer outputs the model's classifications
        self.linear_head = nn.Linear(dimensions, n_classes)

    
    """
    Perform a forward pass (forward propagation) of the model.

    Creates a patch embedding of the image (converting it to a 1D token),
    then encodes the patch's position. The model is then trained.

    Params:
        image: the input image for the model to be trained on

    Returns:
        The inear output layer (tcontains the model's classifications in
        a one-hot encoding)
    """
    def forward(self, image):
        # Get the CUDA hardware device
        device = image.device

        # Get the patch embedding of the image
        x = self.to_patch_embedding(image)
        # Get the positonal embedding of the image, send this embedding to the GPU
        x += self.positional_embedding.to(device, dtype=x.dtype)
        
        # Apply the Transformer network to the model
        x = self.transformer(x)
        # Perform average pooling on the Transformer network
        x = x.mean(dim=1)

        # Apply a skip connection to the model
        x = self.to_latent(x)
        # Output the model's classifications
        return self.linear_head(x)


