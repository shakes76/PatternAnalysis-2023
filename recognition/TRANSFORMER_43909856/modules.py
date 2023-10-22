import os
import os.path as osp
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

""" 
This file will contain the source code of the model's components.
Each component of the model will be implemented as a class or function.
"""

"""
ViT for ImageNet: https://arxiv.org/abs/2205.01580

- ViT has a set of tokens + 1 class token
- Uses an average pooling layer at the end of all convolution layers. 
Generates a 1 dimensional string used for outputting classification of images

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

A masked multi-head attention can be created by adding a mask layer to the 
scaled dot product attention component.
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
