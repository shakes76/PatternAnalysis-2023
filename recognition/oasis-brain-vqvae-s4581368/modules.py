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
                out_channels=hidden_layers//2,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=hidden_layers//2,
                out_channels=hidden_layers,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Conv2d(
                in_channels=hidden_layers,
                out_channels=hidden_layers,
                kernel_size=3,
                stride=1,
                padding=1
            ),
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
    def __init__(self, in_channels, hidden_layers, residual_hidden_layers):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_layers,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            ResidualBlock(
                in_channels=hidden_layers,
                out_channels=hidden_layers,
                residual_hidden_layers=residual_hidden_layers
            ),
            ResidualBlock(
                in_channels=hidden_layers,
                out_channels=hidden_layers,
                residual_hidden_layers=residual_hidden_layers
            ),
            nn.ConvTranspose2d(
                in_channels=hidden_layers,
                out_channels=hidden_layers//2,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=hidden_layers//2,
                out_channels=1, # For Grayscale image
                kernel_size=4, 
                stride=2,
                padding=1
            )
        )

    def forward(self, x):
        return self.layers(x)



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
        distance = torch.sum(self.embedding_layer.weight**2, dim=1) + torch.sum(x_flattened**2, dim=1, keepdim=True) - 2 * torch.matmul(x_flattened, self.embedding_layer.weight.t())

        #codebook_squared = torch.sum(self.embedding_layer.weight**2, dim=1) 
        #input_squared = torch.sum(x_flattened**2, dim=1, keepdim=True)
        #latent_area = 2 * torch.matmul(x_flattened, self.embedding_layer.weight.t())
        #x_distance = input_squared + codebook_squared - latent_area
        x_distance = distance

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
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        x_quantized = x + (x_quantized - x).detach()

        x_quantized = x_quantized.permute(0, 3, 1, 2).contiguous()
        
        return loss, x_quantized, perplexity, encodings

class VQVAE(nn.Module):
    def __init__(self, hidden_layers, hidden_residual_layers, num_embeddings,
                 embedding_dimension, beta):
        super(VQVAE, self).__init__()

        self.encoder = Encoder(
            in_channels=1,
            hidden_layers=hidden_layers,
            residual_hidden_layers=hidden_residual_layers
        )
        
        # Connect encoder output to codebook
        self.conv1 = nn.Conv2d(
            in_channels=hidden_layers,
            out_channels=embedding_dimension,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.vector_quantizer = VectorQuantizer(
            embeddings=num_embeddings,
            embedding_dimensions=embedding_dimension,
            beta=beta
        )

        self.decoder = Decoder(
            in_channels=embedding_dimension,
            hidden_layers=hidden_layers,
            residual_hidden_layers=hidden_residual_layers
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.conv1(x)
        embedding_loss, x_quantized, _, encodings = self.vector_quantizer(x)
        x_reconstructed = self.decoder(x_quantized)
        return embedding_loss, x_reconstructed, x_quantized, encodings

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_class, in_channels, out_channels, kernel_size, padding):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, padding=padding, stride=1)
        self.mask_class = mask_class
        assert mask_class in ['A', 'B'], "Unknown Mask"
        mask = torch.ones(out_channels, in_channels, kernel_size, kernel_size)
        if mask_class == "A":
            mask[:, :, kernel_size // 2, kernel_size // 2:] = 0
            mask[:, :, kernel_size // 2 + 1:] = 0
        else:
            mask[:, :, kernel_size // 2, kernel_size // 2 + 1:] = 0
            mask[:, :, kernel_size // 2] = 0
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class MaskedResidual(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.res_layers = nn.Sequential(
            MaskedConv2d('B', in_channels, in_channels // 2, 1, 0),
            nn.ReLU(True),
            MaskedConv2d('B', in_channels // 2, in_channels // 2, 7, 3),
            nn.ReLU(True),
            MaskedConv2d('B', in_channels // 2, in_channels, 1, 0),
        )

    def forward(self, x):
        return x + self.res_layers(x)
            

class PixelCNN(nn.Module):
    def __init__(self, in_channels, layers, out_channels):
        super(PixelCNN, self).__init__()
        self.in_channels = in_channels
        self.embedded_layers = layers
        self.out_channels = out_channels
        self.model = nn.Sequential(
            MaskedConv2d('A', self.in_channels, self.embedded_layers, 7, 3),
            nn.ReLU(True),
            MaskedResidual(self.embedded_layers),
            nn.ReLU(True),
            #MaskedResidual(self.embedded_layers),
            #nn.ReLU(),
            #nn.BatchNorm2d(self.embedded_layers),
            #MaskedResidual(self.embedded_layers),
            #nn.ReLU(),
            #nn.BatchNorm2d(self.embedded_layers),
            MaskedConv2d("B", self.embedded_layers, self.embedded_layers, 1, 1),
            nn.ReLU(True),
            MaskedConv2d("B", self.embedded_layers, self.embedded_layers, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(self.embedded_layers, self.embedded_layers, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(self.embedded_layers, self.embedded_layers, 4, 2, 1),
        )

    def forward(self, x):
        return self.model(x)

class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()
        self.discriminator = self._make_discriminator()
        self.generator = self._make_generator()

    def _make_generator_block(self, in_planes, planes, stride=2, padding=1):
        layers = []
        layers.append(nn.ConvTranspose2d(in_planes, planes, kernel_size=4, stride=stride, padding=padding, bias=False))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(True))
        return layers
    
    def _make_generator(self):
        blocks = []
        blocks.extend(self._make_generator_block(128, 512, 1, 0))
        blocks.extend(self._make_generator_block(512, 256, 2, 1))
        blocks.extend(self._make_generator_block(256, 128, 2, 1))
        blocks.extend(self._make_generator_block(128, 64, 2, 1))
        blocks.append(nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False))
        blocks.append(nn.Tanh())
        return nn.Sequential(*blocks)


    def _make_discriminator_block(self, in_planes, planes, stride=2):
        layers = []
        layers.append(nn.Conv2d(in_planes, planes, kernel_size=4, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers
    
    def _make_discriminator(self):
        blocks = []
        blocks.extend((self._make_discriminator_block(64, 32, 2)))
        blocks.extend((self._make_discriminator_block(32, 64, 2)))
        blocks.extend((self._make_discriminator_block(64, 128, 2)))
        blocks.extend((self._make_discriminator_block(128, 256, 2)))
        blocks.append(nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, bias=False))
        blocks.append(nn.Flatten())
        blocks.append(nn.Sigmoid())
        return nn.Sequential(*blocks)



