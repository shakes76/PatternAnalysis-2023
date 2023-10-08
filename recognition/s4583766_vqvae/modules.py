"""
Core components of the model required for the pattern recognition task.

Sophie Bates, s4583766.
"""
import torch
import torch.nn as nn

# TODO: extract residual_block to separate module
# TODO: separate encoder and decoder into separate modules

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
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=latent_dim, kernel_size=1, stride=1, padding=0
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


class Decoder(nn.Module):
    """
    Decoder module for the VQ-VAE model.

    From the paper:
    
    The decoder similarly has two residual 3 × 3 blocks, followed by two 
    transposed convolutions with stride 2 and window size 4 × 4. We use the 
    ADAM optimiser [21] with learning rate 2e-4 and evaluate the performance 
    after 250,000 steps with batch-size 128. For VIMCO we use 50 samples in 
    the multi-sample training objective.
    """
    def __init__(self, no_channels, latent_dim):
        super(Decoder, self).__init__()
        self.no_channels = no_channels
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(
            in_channels=no_channels,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.transpose_conv1 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.transpose_conv2 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=no_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.residual_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(out)
        out = self.residual_block(out)
        out = self.residual_block(out)
        out = self.transpose_conv1(out)
        out = self.transpose_conv2(x)
        return out
