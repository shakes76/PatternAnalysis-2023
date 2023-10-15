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
    def __init__(self, in_channels, n_hidden, n_residual):
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
    def __init__(self, dim_in, dim_hidden, dim_residual, n_residuals = 2):
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
    def __init__(self, in_channels, n_hidden, n_residual):
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
        # print('ENCODER CONV1: ', out.shape)

        out = self.relu(out)
        out = self.conv2(out)
        # print('ENCODER CONV2: ', out.shape)

        out = self.residualBlock(out)

        return out
    
class Decoder(nn.Module):
    """
    Decoder
    """
    def __init__(self, in_channels, n_hidden, n_residual, out_channels = 3):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_hidden,
            kernel_size=4,
            stride=2,
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
            out_channels=n_hidden // 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )    

        self.transpose3 = nn.ConvTranspose2d(
            in_channels=n_hidden // 4,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )            

    def forward(self, out):
        out = self.conv1(out)
        out = self.residualBlock(out)
        out = self.transpose1(out)
        out = self.transpose2(out)
        out = self.transpose3(out)
        return out

class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_embeddings : number of embeddings
    - dim_embeddings : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_embeddings, dim_embeddings, beta):
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

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()

        z_flattened = z.view(-1, self.dim_embeddings)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
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

        return loss, z_q, perplexity, min_embeddingsncodings, min_embeddingsncoding_indices

class VQVAE(nn.Module):
    def __init__(self, channels = 3,  n_hidden = 128, n_residual = 32, n_embeddings = 512, dim_embedding = 64, beta = 0.25):
        super(VQVAE, self).__init__()

        self.encoder = Encoder(channels, 
            n_hidden, 
            n_residual
        )
        self.encoder = self.encoder.to(DEVICE)
        
        self.conv = nn.Conv2d(
            in_channels=n_hidden, 
            out_channels=dim_embedding,
            kernel_size=1, 
            stride=1,
            padding=0
        )
        self.conv = self.conv.to(DEVICE)

        self.quantizer = VectorQuantizer(
            n_embeddings, 
            dim_embedding,
            beta
        )
        self.quantizer = self.quantizer.to(DEVICE)
        
        self.decoder = Decoder(
            dim_embedding,
            n_hidden, 
            n_residual,
            channels
        )
        self.decoder = self.decoder.to(DEVICE)

    def forward(self, x):
        # print('VQVAE INPUT: ', x.shape)
        x = self.encoder(x)
        # print('VQVAE ENCODER: ', x.shape)
        x = self.conv(x)
        # print('VQVAE CONV: ', x.shape)
        loss, x_q, perplexity, _, _ = self.quantizer(x)
        print('VQVAE DECODER IN: ', x_q.shape)
        x_hat = self.decoder(x_q)
        # print('VQVAE DECODER: ', x_hat.shape)
        return loss, x_hat, perplexity


"""
|--------------------------------------------------------------------------
| GAN Model for Testing
|--------------------------------------------------------------------------
"""

class Generator(nn.Module):
    def __init__(self, latent_size = 128, channels = 3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            self.layer(latent_size, latent_size * 4, 4, 1, 0),
            self.layer(latent_size * 4, latent_size * 2, 4, 2, 1),
            self.layer(latent_size * 2, latent_size, 4, 2, 1),
            self.layer(latent_size, latent_size // 2, 4, 2, 1),
            nn.ConvTranspose2d(latent_size // 2, channels, 4, 2, 1, bias=False),
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
    def __init__(self, channels = 3, img_size = 64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, img_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            self.layer(img_size, img_size * 2, 4, 2, 1),
            self.layer(img_size * 2, img_size * 4, 4, 2, 1),
            self.layer(img_size * 4, img_size * 8, 4, 2, 1),            
            nn.Conv2d(img_size * 8, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
            nn.Sigmoid()
        )
    
    def layer(self, in_channels, out_channels, kernel=4, stride=1, padding=0, bias=False) :
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

class GAN(nn.Module):
    def __init__(self, channels = 3, latent_dim = 128, img_size = 64):
        super(GAN, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.img_size = img_size

        self.generator = Generator(self.latent_dim)
        self.discriminator = Discriminator(self.channels, self.img_size)

    def forward(self, x):
        return self.generator(x)


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class PixelCNN(nn.Module):
    def __init__(self, input_dim, n_filters=64, kernel_size=7):
        super(PixelCNN, self).__init__()
        self.layers = nn.Sequential(
            MaskedConv2d('A', input_dim, n_filters, kernel_size, 1, kernel_size//2, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(True),
            MaskedConv2d('B', n_filters, n_filters, kernel_size, 1, kernel_size//2, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(True),
            nn.Conv2d(n_filters, 256, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        return F.log_softmax(x, dim=1)