# VQVAE for the OASIS Brain Dataset

import torch

from torch import nn
from torch.nn import funcional as F

class ResidualBlock(nn.Module):
    """
    Residual Block definition for the Encoder and Decoder of the VAE
    """
    def __init__(self, in_channels, out_channels, residual_hidden_layers):
        super(ResidualBlock, self).__init__()
        self._res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=residual_hidden_layers,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=residual_hidden_layers,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias=False
            )
        )
    
    def forward(self, x):
        return x + self._res_block(x)

class Enconder(nn.Module):
    """
    Encoder structer for VAE.
    """
    def __init__(self, in_channels, hidden_layers, downsampling_layers,
                 residual_layers, residual_hidden_layers)


class Decoder(nn.Module):
    pass


