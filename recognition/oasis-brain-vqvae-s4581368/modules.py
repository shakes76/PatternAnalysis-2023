"""
VQVAE for the OASIS Brain Dataset
Ryan Ward
45813685
"""

import torch

from torch import nn
from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class Encoder(nn.Module):
    """
    Encoder structure for VAE.
    """
    def __init__(self, in_channels, hidden_layers, residual_hidden_layers):
        super(Encoder, self).__init__()

        # Common encoder parameters
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_layers,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=hidden_layers,
                out_channels=hidden_layers,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(True),
            ResidualBlock(
                in_channels=hidden_layers,
                out_channels=hidden_layers,
                residual_hidden_layers=residual_hidden_layers
            ),
            ResidualBlock(
                in_channels=hidden_layers,
                out_channels=hidden_layers,
                residual_hidden_layers=residual_hidden_layers
            )
        )
    
    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    """
    Decoder structure for VAE
    """
    pass

class VectorQuantizer(nn.Module):
    """
    Vector Quantized layer
    """
    def __init__(self, embeddings, embedding_dimensions, beta):
        super(VectorQuantizer, self).__init__()
        self.num_e = embeddings
        self.num_e_dim = embedding_dimensions
        self.beta = beta

        self.embedding_layer = nn.Embedding(embeddings, embedding_dimensions)
        self.embedding_layer.weight.data.uniform_(-1/embeddings, 1/embedding_dimensions)

    def forward(self, x: torch.Tensor):
        """
        TODO: Document me
        """
        # Convert input tensor from BCHW to BHWC
        x = x.permute(0, 2, 3, 1).contiguous()
        x_shape = x.shape

        # Flatten to snap to codebook
        x_flattened = x.view(-1, self.num_e_dim)

        # Calulate distance from input to codebook mapping
        # Insert Equation
        codebook_squared= torch.sum(self.embedding_layer.weight**2, dim=1) 
        input_squared = torch.sum(x_flattened**2, dim=1, keepdim=True)
        latent_area = 2 * torch.matmul(x_flattened, self.embedding_layer.weight.t())
        x_distance = input_squared + codebook_squared - latent_area


        encoding_idx = torch.argmin(x_distance, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_idx.shape[0], self.num_e).to(device)
        encodings.scatter_(1, encoding_idx, 1)

        # Obtain the (quantized) latent vectors
        x_quantized = torch.matmul(encodings, self.embedding_layer.weight).view(x_shape)
        
        # Find losses
        embedding_loss = F.mse_loss(x_quantized.detach(), x)
        quantization_loss = F.mse_loss(x_quantized, x.detach())
        
        loss = quantization_loss + (self.beta * embedding_loss)

        # Perplexity - Do I need?
        e_mean = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e10)))

        x_quantized = x + (x_quantized - x).detach()

        x_quantized = x_quantized.permute(0, 3, 1, 2).contiguous()
        
        return loss, x_quantized, perplexity, encodings



        
        
        



