import torch
import torch.nn as nn

class PositionalImageEmbedding(nn.Module):
    def __init__(self, input_channels, embed_dim, bands=4):
        """
        Initialise the PositionalImageEmbedding module.

        Params:
            input_channels (int): Number of input channels in the image.
            embed_dim (int): Dimension of the embedded image representation.
            bands (int): Number of Fourier feature bands for positional encoding.
        """
        super().__init__()

        # Initialise the Fourier features for positional encoding
        self.ff = self.fourier_features(bands)

        # Initialise a 1D convolution layer to process the concatenated input
        self.conv = nn.Conv1d(input_channels + self.ff.shape[0], embed_dim, 1)

    def forward(self, x):
        """
        Forward pass of the PositionalImageEmbedding module.

        Args:
            x (Tensor): Input image tensor of shape (BATCH_SIZE x CHANNELS x HEIGHT x WIDTH).

        Returns:
            (Tensor): Embedded image representation of shape (HEIGHT*WIDTH x BATCH_SIZE x EMBED_DIM).
        """
        # Create position encoding with the same shape as the input
        enc = self.ff.unsqueeze(0).expand(x.shape[0], -1, -1).type_as(x)

        # Concatenate the position encoding along the channel dimension
        x = torch.cat([x, enc], dim=1)

        # Flatten the input
        x = x.flatten(2)

        # Apply 1D convolution
        x = self.conv(x)

        # Permute the dimensions for the final output
        x = x.permute(2, 0, 1)

        return x
