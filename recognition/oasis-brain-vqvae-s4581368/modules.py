"""
VQVAE for the OASIS Brain Dataset
Ryan Ward
45813685
"""
import torch
from torch import nn
from torch.nn import functional as F

"""Global Device"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResidualBlock(nn.Module):
    """
    Residual Block definition for the Encoder and Decoder of the VAE
    :param int in_channels: The number of input channels for the ResidualBlock
    :param int out_channels: The number of output channels for the ResidualBlock
    :param int residual_hidden_layers: The number of hidden layers in the ResidualBlock

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
        """
        Forward pass throught the residual block
        :param Tensor x: The Tensor input
        """
        return x + self._res_block(x)


class Encoder(nn.Module):
    """
    Encoder structure for VAE.
    :param int in_channels: The number of input channels for the Encoder
    :param int out_channels: The number of output channels for the Encoder
    :param int residual_hidden_layers: The number of hidden layers in the Encoder
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
        """
        Forward pass of the encoder
        :param Tensor x: The Tensor input
        """
        return self.layers(x)


class Decoder(nn.Module):
    """
    Decoder structure for VAE
    :param int in_channels: The number of input channels for the Decoder 
    :param int out_channels: The number of output channels for the Decoder   
    :param int residual_hidden_layers: The number of hidden layers in the Decoder 
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
        """
        Forward pass of the decoder
        :param Tensor x: The Tensor input
        """
        return self.layers(x)


class VectorQuantizer(nn.Module):
    """
    Vector Quantized layer
    :param int embeddings: The number of embeddings in the quantized layer
    :param int embedding_dimensions: The number of embedding dimensions in the quantized layer
    :param float beta: The commitment cost as defined in the original paper
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
        Forward pass of the Vector Quantized layer
        :param Tensor x: The Tensor input
        """
        # Convert input tensor from BCHW to BHWC
        x = x.permute(0, 2, 3, 1).contiguous()
        x_shape = x.shape

        # Flatten to snap to codebook
        x_flattened = x.view(-1, self.num_e_dim)

        # Calulate distance from input to codebook mapping
        codebook_squared = torch.sum(self.embedding_layer.weight**2, dim=1) 
        input_squared = torch.sum(x_flattened**2, dim=1, keepdim=True)
        latent_area = 2 * torch.matmul(x_flattened, self.embedding_layer.weight.t())
        x_distance = input_squared + codebook_squared - latent_area

        # Calculate nearest codebook indice
        encoding_idx = torch.argmin(x_distance, dim=1).unsqueeze(1)

        # Update the encodings
        encodings = torch.zeros(encoding_idx.shape[0], self.num_e).to(device)
        encodings.scatter_(1, encoding_idx, 1)

        # Obtain the (quantized) latent vectors
        x_quantized = torch.matmul(encodings, self.embedding_layer.weight).view(x_shape)
        
        # Find losses
        embedding_loss = F.mse_loss(x_quantized.detach(), x)
        quantization_loss = F.mse_loss(x_quantized, x.detach())
       
        # Calculate the total loss, see loss equation in README.md
        loss = quantization_loss + (self.beta * embedding_loss)

        # Find the quantized representation of the inputs
        x_quantized = x + (x_quantized - x).detach()

        # Reshape the quantized image
        x_quantized = x_quantized.permute(0, 3, 1, 2).contiguous()
        
        return loss, x_quantized, embedding_loss, encodings

class VQVAE(nn.Module):
    """
    The VQ-VAE top level module
    :param int hidden_layers: The number of hidden layers in the VQ-VAE
    :param int hidden_residual_layers: The number of hidden layers in the ResidualBlocks
    :param int num_embeddings: The number of codebook indices
    :param int embedding_dimensions: The dimension of the vector quantized layer
    :param float beta: The commitment cost (see original paper)
    """
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
        """
        Forward pass of the VQ-VAE
        :param Tensor x: The Tensor input
        :returns float embedding_loss: The loss associated with 'snapping' the input to a codebook indice
        :returns Tensor x_reconstructed: Reconstructed input images
        :returns Tensor x_quantized: Quantized input images
        :returns Tensor x_quantized: The codebook encodings
        """
        x = self.encoder(x)
        x = self.conv1(x)
        embedding_loss, x_quantized, _, encodings = self.vector_quantizer(x)
        x_reconstructed = self.decoder(x_quantized)
        return embedding_loss, x_reconstructed, x_quantized, encodings

class GAN(nn.Module):
    """
    Simple yet effective GAN for generating original discrete representation 
    :param: None
    """ 
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
        blocks.extend(self._make_generator_block(512, 512, 2, 1))
        blocks.extend(self._make_generator_block(512, 256, 2, 1))
        blocks.extend(self._make_generator_block(256, 128, 2, 1))
        blocks.append(nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False))
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
        blocks.extend((self._make_discriminator_block(128, 32, 2)))
        blocks.extend((self._make_discriminator_block(32, 64, 2)))
        blocks.extend((self._make_discriminator_block(64, 128, 2)))
        blocks.extend((self._make_discriminator_block(128, 256, 2)))
        blocks.append(nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, bias=False))
        blocks.append(nn.Flatten())
        blocks.append(nn.Sigmoid())
        return nn.Sequential(*blocks)



