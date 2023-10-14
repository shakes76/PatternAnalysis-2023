'''
contains the source code of the components of the model. Each component is
implementated as a class or a function
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

class PatchEmbedding(nn.Module):
    """
    Takes a 2D image and turns it into an input for the tranformer by flattening it
    into a 1D sequence learnable embedding vector as described in the paper 

    Args:
        in_channels (int): Number of colour channels. In the case of the ADNC, there are 3 (RGB)
            -> default = 3

        patch_size (int): Number of patches each image will be converted into. Must be a square number
            -> default = 16
        
        embedding_dim (int): Size of embedding vector for each patch
            -> default = 768
    """

    def __init__(self,
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768):
        super().__init__()

        # Layer which converts an image into patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        # Flattens the patch feature maps to 1D 
        # Only flatten the feature map dimension
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    # Forward Method
    def forward(self, x):
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        return x_flattened.permute(0,2,1) # such that the embedding is on the final dimension