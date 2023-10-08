"""
Core components of the model required for the pattern recognition task.

Sophie Bates, s4583766.
"""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder module for the VQ-VAE model.

    The encoder consists of 2 strided convolutional layers with stride 2 and 
    window size 4 × 4, followed by two residual 3 × 3 blocks (implemented as 
    ReLU, 3x3 conv, ReLU, 1x1 conv), all having 256 hidden units.
    """

    def __init__(self, no_channels, latent_dim):
        super(Encoder, self).__init__()
        self.no_channels = no_channels
        # self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(
            in_channels=no_channels,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1
        )

        self.residual_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.residual_block(out)
        out = self.residual_block(out)
        out = self.conv3(out)
        return out
