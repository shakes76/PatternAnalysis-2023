import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    """Split images into patches
    """
    def __init__(self, img_size: int, patch_size:int, embed_dim=768, in_channels=1, linear_mode=False):
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
                # TODO: add layerNorm?
                nn.Linear(self.patch_size * self.patch_size * in_channels, embed_dim)
                # TODO: add LayerNorm?
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