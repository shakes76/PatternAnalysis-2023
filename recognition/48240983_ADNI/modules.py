"""
Created on Thursday 19th Oct
Alzheimer's disease using PyTorch (ViT Transformer)
This code outlines a structure for a computer vision model for classifying images. 
It consists of two modules: a VisionModel module that combines embedded data with positional encodings 
and uses a stack of transformer encoder layers for processing;
the Embedder module uses convolution to turn image patches into embeddings. 

@author: Gaurika Diwan
@ID: s48240983
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#  Embedder module defined 
class Embedder(nn.Module):
    def __init__(self, patch_size, dim):
        """
        the Embedder module defined.

        Args:
            patch_size (int): The size of the image patches.
            dim (int): The dimension of the embedding.

        """
        super(Embedder, self).__init__()

# An encoder using Conv2d and BatchNorm2d
        self.encoder = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(dim)
        )

    def forward(self, inputs):
        """
        Forward pass of the Embedder module.

        Args:
            inputs (torch.Tensor): Input image data.

        Returns:
            torch.Tensor: Flattened embeddings.

        """
        encoded = self.encoder(inputs)
        flattened = encoded.view(encoded.size(0), encoded.size(1), -1)
        return flattened
    
#  VisionModel module defined
class VisionModel(nn.Module):
    def __init__(self, num_classes, image_size, patch_size, num_patches, dim, depth, num_heads, mlp_dim):
        super(VisionModel, self).__init__()
        """
        Defined the VisionModel.

        Args:
            num_classes (int): Number of output classes.
            image_size (tuple): The size of the input image (height, width).
            patch_size (int): The size of image patches.
            num_patches (int): The total number of patches.
            dim (int): The dimension of the embedding.
            depth (int): The number of transformer encoder layers.
            num_heads (int): The number of attention heads in the transformer.
            mlp_dim (int): The dimension of the MLP head.

        """
        # Embedder called to convert image patches into embeddings

        self.embedder = Embedder(patch_size, dim)
        self.token = nn.Parameter(torch.randn(1, 1, dim))
        self.positions = nn.Parameter(torch.randn(1, num_patches + 1, dim))

       # Create a transformer layer for processing the embeddings
        self.transformer = nn.Transformer(
            d_model=dim,
            nhead=num_heads,
            num_encoder_layers=depth,
            dim_feedforward=mlp_dim,
            dropout=0.1
        )
       # Create a multi-layer perceptron  head for classification
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, inputs):
        """
        Forward pass of the VisionModel.

        Args:
            inputs : Input image data.

        Returns:
            torch.Tensor: Model predictions.

        """
        embedded = self.embedder(inputs)

        #Token embedding and place it with the embedded patches
        token = self.token.repeat(embedded.size(0), 1, 1)
        embedded = torch.cat([token, embedded], dim=1)
        embedded += self.positions

        encoded = self.transformer(embedded)
        aggregated = torch.mean(encoded, dim=1)
        # Final classification through the MLP head

        return self.mlp_head(aggregated)

# Define hyperparameters

num_classes = 10
image_size = (150, 150)
patch_size = 16
num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
dim = 64
depth = 12
num_heads = 8
mlp_dim = 256

# Instance of the VisionModel created

model = VisionModel(num_classes, image_size, patch_size, num_patches, dim, depth, num_heads, mlp_dim)
