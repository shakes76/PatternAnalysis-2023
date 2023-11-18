##################################
#
# Author: Joshua Wallace
# SID: 45809978
#
###################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import DEVICE

"""
|--------------------------------------------------------------------------
| VQVAE Model Modules
|--------------------------------------------------------------------------
"""

class ResidualLayer(nn.Module):
    """
    Layer for the residual block.
    """
    def __init__(self, in_channels, n_hidden, n_residual):
        """
        Initialize the residual layer.

        Parameters
        ----------
        param1 : in_channels
            Number of input channels.
        param2 : n_hidden
            Number of hidden layers.
        param3: n_residual
            Number of residual hidden layers.
        """
        
        super(ResidualLayer, self).__init__()

        self.residual = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, 
                out_channels=n_residual, 
                kernel_size=3, 
                stride=1, 
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_residual, 
                out_channels=n_hidden, 
                kernel_size=1, 
                stride=1, 
                padding=0
            ),
        )
    
    def forward(self, out):
        return out + self.residual(out)

class ResidualBlock(nn.Module) :
    """
    Residual stack to pass the output deeper into the network.
    """
    def __init__(self, dim_in, dim_hidden, dim_residual, n_residuals = 2):
        """
        Initialize the residual layer.

        Parameters
        ----------
        param1 : dim_in
            Input dimension
        param2 : dim_hidden
            Dimension of hidden layers.
        param3: dim_residual
            Dimension of residual hidden layers.
        param4: n_residuals
            Number of residual layers.
        """
                
        super(ResidualBlock, self).__init__()
        self.n_residuals = n_residuals
        self.seq = nn.ModuleList(
            [ResidualLayer(dim_in, dim_hidden, dim_residual)] * n_residuals)

    def forward(self, x):
        for layer in self.seq:
            x = layer(x)
        x = F.relu(x)
        return x

class Encoder(nn.Module):
    """
    Encoder accepts an image of size (B, C, H, W) and returns a tensor of size (B, n_hidden, H/2^2, W/2^2).
    It consists of 2 convolutional layers with stride 2, kernel 4x4, succeeded 1 residual block.
    """
    def __init__(self, in_channels, n_hidden, n_residual):
        """
        Initialize the encoder.

        Parameters
        ----------
        param1 : in_channels
            Number of input channels (image channels).
        param2 : n_hidden
            Number of hidden layers for the VQVAE.
        param3: n_residual
            Number of residual hidden layers for the VQVAE.
        """
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_hidden // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            in_channels=n_hidden // 2, 
            out_channels=n_hidden, 
            kernel_size=4, 
            stride=2, 
            padding=1
        )

        self.residualBlock = ResidualBlock(
            n_hidden, 
            n_hidden, 
            n_residual, 
            2
        )
        
        self.relu = nn.ReLU()

    def forward(self, out):
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.residualBlock(out)

        return out
    
class Decoder(nn.Module):
    """
    Decoder accepts an encoded image of size (B, n_hidden, H/2^2, W/2^2) and returns a tensor of size (B, C, H, W).
    It consists of a convolutional layer on the input, then a residual block, succeeded by three layers of transposed convolution with stride 2, kernel 4x4.
    """
    def __init__(self, in_channels, n_hidden, n_residual, out_channels = 3):
        """
        Initialize the decoder.

        Parameters
        ----------
        param1 : in_channels
            Number of input channels.
        param2 : n_hidden
            Number of hidden layers for the VQVAE.
        param3: n_residual
            Number of residual hidden layers for the VQVAE.
        param: out_channels
            Number of channels for the output (should be equal to number of image channels)
        """
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.residualBlock = ResidualBlock(
            n_hidden, 
            n_hidden, 
            n_residual, 
            2
        )

        self.transpose1 = nn.ConvTranspose2d(
            in_channels=n_hidden,
            out_channels=n_hidden // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        
        self.transpose2 = nn.ConvTranspose2d(
            in_channels=n_hidden // 2,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )    

        # self.transpose3 = nn.ConvTranspose2d(
        #     in_channels=n_hidden // 4,
        #     out_channels=out_channels,
        #     kernel_size=4,
        #     stride=2,
        #     padding=1,
        # )            

    def forward(self, out):
        out = self.conv1(out)
        out = self.residualBlock(out)
        out = self.transpose1(out)
        out = self.transpose2(out)
        # out = self.transpose3(out)
        return out

class VectorQuantizer(nn.Module):
    """
    The Vector Quantizer quantizes the encoded input tensor z to a discrete latent representation z_q.
    """

    def __init__(self, n_embeddings, dim_embeddings, beta):
        """
        Initialize the encoder.

        Parameters
        ----------
        param1 : n_embeddings
            Number of embeddings in the codebook.
        param2 : dim_embedding
            Dimension of each embedding.
        param3: beta
            Commitment cost term for the loss of the vector quantizer.
        """
        super(VectorQuantizer, self).__init__()
        self.n_embeddings = n_embeddings
        self.dim_embeddings = dim_embeddings
        self.beta = beta

        self.embedding = nn.Embedding(
            self.n_embeddings, 
            self.dim_embeddings
        )
        self.embedding.weight.data.uniform_(-1.0 / self.n_embeddings, 1.0 / self.n_embeddings)

        

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        
        Taken directly from: //github.com/MishaLaskin/vqvae/blob/master/models/vqvae.py
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()

        z_flattened = z.view(-1, self.dim_embeddings)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        codebook = torch.argmin(d, dim=1)
        min_embeddingsncoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_embeddingsncodings = torch.zeros(
            min_embeddingsncoding_indices.shape[0], self.n_embeddings).to(DEVICE)
        min_embeddingsncodings.scatter_(1, min_embeddingsncoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_embeddingsncodings, self.embedding.weight).view(z.shape)
        # print('VQ MATMUL: ', z_q.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()
        # print('VQ GRADIENTS: ', z_q.shape)

        # perplexity
        e_mean = torch.mean(min_embeddingsncodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        # print('VQ PERMUTE: ', z_q.shape)

        return loss, z_q, perplexity, min_embeddingsncodings, codebook

class VQVAE(nn.Module):
    """
    VQ-VAE class, which accepts an image as input then trains the encoder, VQ and decoder accordingly.
    """
    def __init__(self, channels = 3,  n_hidden = 128, n_residual = 32, n_embeddings = 512, dim_embedding = 64, beta = 0.25):
        """
        Initialize the encoder.

        Parameters
        ----------
        param1 : channels, 3
            Number of input channels (image channels).
        param2 : n_hidden, 128
            Number of hidden layers for the VQVAE.
        param3: n_residual, 32
            Number of residual hidden layers for the VQVAE.
        param4: n_embeddings, 512
            Number of embeddings in the codebook.
        param5: dim_embedding, 64
            Dimension of each embedding.
        param6: beta, 0.25
            Commitment cost term for the loss of the vector quantizer.
        """
        super(VQVAE, self).__init__()

        self.encoder = Encoder(channels, 
            n_hidden, 
            n_residual
        )
        
        self.conv = nn.Conv2d(
            in_channels=n_hidden, 
            out_channels=dim_embedding,
            kernel_size=1, 
            stride=1,
        )

        self.quantizer = VectorQuantizer(
            n_embeddings, 
            dim_embedding,
            beta
        )
        
        self.decoder = Decoder(
            dim_embedding,
            n_hidden, 
            n_residual,
            channels
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.conv(x)
        loss, x_q, perplexity, _, _ = self.quantizer(x)
        x_hat = self.decoder(x_q)
        return loss, x_hat, perplexity


"""
|--------------------------------------------------------------------------
| GAN Model for Testing
|--------------------------------------------------------------------------
"""

class Generator(nn.Module):
    """
    Generator accepts a latent vector of size (B, latent_size, 1, 1) and returns an image of size (B, channels, H, W).
    """
    def __init__(self, latent_size = 128, size=64, channels = 3):
        """
        Initialize the generator.

        Parameters
        ----------
        param1 : latent_size
            Latent size of the input vector.
        param2 : channels
            Number of channels for the image.
        """
        self.noise = latent_size
        self.size = size

        super(Generator, self).__init__()
        self.main = nn.Sequential(
            self.layer(self.noise, self.size * 8, 4, 1, 0),
            self.layer(self.size * 8, self.size * 4, 4, 2, 1),
            self.layer(self.size * 4, self.size * 2, 4, 2, 1),
            self.layer(self.size * 2, self.size, 4, 2, 1),
            nn.ConvTranspose2d(self.size, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def layer(self, in_channels, out_channels, kernel=4, stride=1, padding=0, bias=False) :
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    """
    Discriminator accepts an image of size (B, channels, H, W) and returns a probability of the image being real.
    """
    def __init__(self, channels = 3, img_size = 64):
        """
        Initialize the discriminator.

        Parameters
        ----------
        param1 : channels
            Number of input channels (image channels).
        param2 : img_size, 64
            Size of the image
        """
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, img_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            self.layer(img_size, img_size * 2, 4, 2, 1),
            self.layer(img_size * 2, img_size * 4, 4, 2, 1),
            self.layer(img_size * 4, img_size * 8, 4, 2, 1),            
            nn.Conv2d(img_size * 8, 1, 4, 2, 0, bias=False),
            nn.Sigmoid()
        )
    
    def layer(self, in_channels, out_channels, kernel=4, stride=1, padding=0, bias=False) :
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        return self.main(input)

class GAN(nn.Module):
    def __init__(self, channels = 3, latent_dim = 128, img_size = 64):
        super(GAN, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.img_size = img_size

        self.generator = Generator(self.latent_dim, self.img_size, self.channels)
        self.discriminator = Discriminator(self.channels, self.img_size)

    def forward(self, x):
        return self.generator(x)