"""
Script to create a Visual Transformer model

@author: Rodger Xiang s4642506
"""

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size:int, embed_dim:int, in_channels:int):
        """
        initialise patch embedding layer for the ViT

        Args:
            patch_size (int):   H or W of patch size (should be a square number)
            embed_dim (int):    Size of patch embedding stays constant across entire network
            in_channels (int):  RGB channel of image
        """
        super().__init__()        
        # embedding using linear layers
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                        p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim),
        )
        
    def forward(self, x):
        return self.projection(x)
    
    
class ViT(nn.Module):
    def __init__(self, img_size:int,
                    patch_size:int,
                    img_channels:int,
                    num_classes:int,
                    embed_dim:int,
                    depth:int,
                    num_heads:int,
                    mlp_dim:int,
                    drop_prob=0.1):
        """Creates a visual transformer model

        Args:
            img_size (int):     size of images to be passed into the model
            patch_size (int):   size of a image patch
            img_channels (int): number of colour channels of the image
            num_classes (int):  number of classes of the ADNI dataset
            embed_dim (int):    dimension of the patch embedding
            depth (int):        number of transformer encoder layers
            num_heads (int):    number of attention heads for each encoder layer
            mlp_dim (int):      number of hidden units of the feed forward layer in the encoder layers
            drop_prob (float, optional): probability for dropout in the model. Defaults to 0.1.
        """
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = PatchEmbedding(patch_size=patch_size,
                                            embed_dim=embed_dim,
                                            in_channels=img_channels)
        #class token to determine which class the image belongs to
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) #zeros or randn
        #positional information of patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
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