"""
Created on Wednesday September 20 2023

@author: Rodger Xiang s4642506
"""

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    """Split images into patches
    """
    def __init__(self, img_size: int, patch_size:int, embed_dim=768, in_channels=1, linear_mode=True):
        """
        initialise patch embedding layer for the ViT

        Args:
            img_size (int): H or W of image (should be a square number)
            patch_size (int): H or W of patch size (should be a square number)
            embed_dim (int, optional): Size of patch embedding stays constant across entire network. Defaults to 768.
            in_channels (int, optional): RGB channel of image. Defaults to 1 for greyscale.
            linear_mode (bool): whether to use linear projection method or convolution
        """
        super().__init__()        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.linear_mode = linear_mode
        #kernel is the same size of the patch size and will never overlap since
        self.projection = nn.Conv2d(in_channels=in_channels, 
                                    out_channels=embed_dim,
                                    kernel_size=self.patch_size,
                                    stride=self.patch_size)
        
        # embedding using linear layers
        if self.linear_mode:
            self.projection = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                            p1=self.patch_size, p2=self.patch_size),
                # nn.LayerNorm(self.patch_size * self.patch_size * in_channels),
                nn.Linear(self.patch_size * self.patch_size * in_channels, embed_dim),
                # nn.LayerNorm(embed_dim)
            )
        
    def forward(self, x):
        """forward pass

        Args:
            x (tensor): image tensor of (batch_size, img_channels, H, W)
        
        Returns:
            tensor: (batch_size, num_patches, embed_dim)
        """
        x = self.projection(x)
        
        #if not using the linear project we need to reshape the tensor after convolution
        if not self.linear_mode:
            x = x.flatten(2).transpose(1, 2)
        return x
    
    
class ViT_torch(nn.Module):
    def __init__(self, img_size:int,
                    patch_size:int,
                    img_channels:int,
                    num_classes:int,
                    embed_dim:int,
                    depth:int,
                    num_heads:int,
                    mlp_dim:int,
                    drop_prob=0.1,
                    linear_embed=False):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size=img_size,
                                            patch_size=patch_size,
                                            embed_dim=embed_dim,
                                            in_channels=img_channels,
                                            linear_mode=linear_embed) #TODO: try changing to linear mode
        #class token to determine which class the image belongs to
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) #zeros or randn
        #positional information of patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_prob)
        
        #transform encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                    nhead=num_heads,
                                                    dim_feedforward=mlp_dim,
                                                    activation='gelu',
                                                    batch_first=True,
                                                    norm_first=True, 
                                                    layer_norm_eps=1e-6)
        
        self.transform_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                                        num_layers=depth)
        
        # self.norm_layer = nn.LayerNorm(embed_dim)
        self.latent = nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        num_batch = x.shape[0]
        # convert images into patches
        x = self.patch_embed(x)
        # concate the class token
        class_token = self.class_token.expand(num_batch, -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        #pass x through encoders
        x = self.transform_encoder(x)
        
        #get only the class token for output
        output = x[:, 0]
        # output = self.norm_layer(output)
        output = self.latent(output)
        output = self.head(output)
        return output