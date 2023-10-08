"""
Created on Monday Sep 18 12:20:00 2023

This script is for building the components of the VQVAE model. 
The model is implemented as a class that will be called when training.

@author: Gabriel Russell
@ID: s4640776

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence


"""
Define the hyperparameters used for the model.
"""
class Parameters():
    def __init__(self):
        self.batch_size = 32 #128
        self.num_training_updates = 15000
        self.num_hiddens = 128
        self.num_residual_hiddens = 32
        self.embedding_dim = 128
        self.num_embeddings = 512
        self.commitment_cost = 0.25
        self.learn_rate = 2e-4
        self.grey_channel = 1
        self.features = 128
        self.channel_noise = 100

    
"""
Residual layer containing [ReLU, 3x3 conv, ReLU, 1x1 conv]
"""
class Residual_layer(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual_layer, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(inplace = False),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace = False),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


"""
Creates a Residual block consisting of 2 residual layers
"""
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(ResidualBlock, self).__init__()
        self.layer_1 = Residual_layer(in_channels, num_hiddens, num_residual_hiddens)
        self.layer_2 = Residual_layer(in_channels, num_hiddens, num_residual_hiddens)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = F.relu(x)
        return x
    
    
"""
Encoder class which consists of 2 strided convolutional layers 
with stride 2 and kernel size 4x4, followed by a residual block.
"""
class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Encoder, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        
        self.conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)

        self.residual_block = ResidualBlock(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        inputs = self.conv_1(inputs)
        inputs = F.relu(inputs)
        
        inputs = self.conv_2(inputs)
        inputs = F.relu(inputs)
        
        inputs = self.residual_block(inputs)
        return inputs
    

"""
Decoder consists of a residual block, followed by 2 transposed convolutions 
with stride 2 and kernel size 4x4.
"""    
class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Decoder, self).__init__()
        p = Parameters()

        self.residual_block = ResidualBlock(in_channels=in_channels,
                                             num_hiddens=num_hiddens,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self.transposed_conv_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self.transposed_conv_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=p.grey_channel,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        inputs = self.residual_block(inputs)
        
        inputs = self.transposed_conv_1(inputs)
        inputs = F.relu(inputs)

        inputs = self.transposed_conv_2(inputs)
        return inputs
    
""" 
The Vector Quantizer layer quantizes the input tensor.
BCHW (Batch, Channel, Height, Width) tensor is converted to BHWC shape.
Reshaped into [B*H*W, C] and all other dimensions are flattened.
"""
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

    
"""
Class that build model using other classes such as Encoder, 
VectorQuantizer and Decoder. Feeds in the hyperparameters that 
are initialised above. 
"""
class VQVAEModel(nn.Module):
    def __init__(self):
        super(VQVAEModel, self).__init__()
        p = Parameters()
        self.encoder = Encoder(p.grey_channel, p.num_hiddens, 
                                p.num_residual_hiddens)
        self.conv_layer = nn.Conv2d(in_channels=p.num_hiddens, 
                                      out_channels=p.embedding_dim,
                                      kernel_size=1, 
                                      stride=1)
        
        self.quantizer = VectorQuantizer(p.num_embeddings, p.embedding_dim,
                                           p.commitment_cost)
        self.decoder = Decoder(p.embedding_dim,
                                p.num_hiddens, 
                                p.num_residual_hiddens)

    def forward(self, x):
        x = self.encoder(x)
        x = self.conv_layer(x)
        loss, quantized, perplexity, _ = self.quantizer(x)
        x_recon = self.decoder(quantized)

        return loss, x_recon, perplexity

"""
Class for initialising an Adam Optimizer for VQVAE model
"""
class Optimizer():
    def __init__(self, model):
        self.model = model
        p = Parameters()
        self.Adam = optim.Adam(self.model.parameters(), p.learn_rate)

"""
Building a DCGAN to generate images from trained images outputted by VQVAE
"""

"""
Discriminator Class of DCGAN
"""
class Discriminator(nn.Module):
    def __init__(self, grey_channel, features):
        super(Discriminator, self).__init__()
        #64 x 64 input image
        self.input = nn.Sequential(
            nn.Conv2d(grey_channel, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(features)
        )

        #Conv Block 1
        self.block_1 = nn.Sequential(
            nn.Conv2d(features, features*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*2),
            nn.LeakyReLU(0.2)
        )

        #Conv Block 2
        self.block_2 = nn.Sequential(
            nn.Conv2d(features*2, features*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*4),
            nn.LeakyReLU(0.2)
        )

        #Conv Block 3
        self.block_3 = nn.Sequential(
            nn.Conv2d(features*4, features*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*8),
            nn.LeakyReLU(0.2)
        )

        #Output - reduces to 1 dimension
        self.output = nn.Sequential(
            nn.Conv2d(features*8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        input = self.input(x)
        conv = self.block_1(input)
        conv = self.block_2(conv)
        conv = self.block_3(conv)
        output = self.output(conv)
        return output
    

"""
Generator Class of DCGAN
"""
class Generator(nn.Module):
    def __init__(self, channel_noise, channels, features):
        super(Generator, self).__init__()    
        #Input block for noise 
        self.input = nn.Sequential(
            nn.ConvTranspose2d(channel_noise, features*16, kernel_size=4, stride=2, padding=0).
            nn.BatchNorm2d(features*16),
            nn.ReLU()
        )

        #Conv transpose block 1
        self.block_1 = nn.Sequential(
            nn.ConvTranspose2d(features*16, features*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*8),
            nn.ReLU()
        )

        #Conv transpose block 2
        self.block_2 = nn.Sequential(
            nn.ConvTranspose2d(features*8, features*4, kernel_size=4, stride=2, padding=1),
             nn.BatchNorm2d(features*4),
            nn.ReLU()
        )

        #Conv transpose block 3
        self.block_3 = nn.Sequential(
            nn.ConvTranspose2d(features*4, features*2, kernel_size=4, stride=2, padding=1),
             nn.BatchNorm2d(features*2),
            nn.ReLU()
        )

        #Conv transpose block 4
        self.block_4 = nn.Sequential(
            nn.ConvTranspose2d(features*2, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        input = self.input(x)
        conv_transpose = self.block_1(input)
        conv_transpose = self.block_2(conv_transpose)
        conv_transpose = self.block_3(conv_transpose)
        output = self.block_4(conv_transpose)
        return output


