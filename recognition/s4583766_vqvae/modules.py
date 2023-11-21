"""
Core components of the model required for the pattern recognition task.

Author: Sophie Bates, s4583766.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResidualBlock(nn.Module):
    """Residual block for the VQ-VAE Encoder and Decoder models."""
    def __init__(self, n_channels_in,n_channels_out, n_residual_hidden_layers):
        super(ResidualBlock, self).__init__()
        self._residual_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=n_channels_in, 
                      out_channels=n_residual_hidden_layers, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1, 
                      bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_residual_hidden_layers, 
                      out_channels=n_channels_out, 
                      kernel_size=1, 
                      stride=1, 
                      padding=0, 
                      bias=False),
        )
    def forward(self, x):
        return x + self._residual_block(x)

class Encoder(nn.Module):
    """
    Encoder module for the VQ-VAE model.

    The encoder takes an input as an input and outputs a latent vector z_e(x), 
    that will be fed into the VectorQuantizer.
    """

    def __init__(self, n_inputs, n_hidden_layers,n_residual_hidden_layers):
        super(Encoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=n_inputs,
                out_channels=n_hidden_layers//2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=n_hidden_layers//2, 
                out_channels=n_hidden_layers, 
                kernel_size=4, 
                stride=2, 
                padding=1
            ),
            ResidualBlock(
                n_channels_in=n_hidden_layers, 
                n_channels_out=n_hidden_layers, 
                n_residual_hidden_layers=n_residual_hidden_layers
            ),
            ResidualBlock(n_channels_in=n_hidden_layers, 
                          n_channels_out=n_hidden_layers, 
                          n_residual_hidden_layers=n_residual_hidden_layers
            ),
        )

    def forward(self, x):
        return self.layers(x)
class Decoder(nn.Module):
    """
    Decoder module for the VQ-VAE model. Takes discrete latent vectors as
    input and outputs a reconstruction of the original image.

    The model size was taken from the paper: (https://arxiv.org/pdf/1711.00937.pdf):
    """
    def __init__(self, n_inputs, n_hidden_layers, n_residual_hidden_layers): #TODO: remove n_residual_layers?
        super(Decoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=n_inputs,
                out_channels=n_hidden_layers,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ResidualBlock(
                n_channels_in=n_hidden_layers,
                n_channels_out=n_hidden_layers,
                n_residual_hidden_layers=n_residual_hidden_layers
            ),
            ResidualBlock(
                n_channels_in=n_hidden_layers,
                n_channels_out=n_hidden_layers,
                n_residual_hidden_layers=n_residual_hidden_layers
            ),
            nn.ConvTranspose2d(
                in_channels=n_hidden_layers,
                out_channels=n_hidden_layers//2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=n_hidden_layers//2,
                out_channels=1, #TODO: set this no. channels as a constant in driver
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )

    def forward(self, x):
        return self.layers(x)

class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck module for the VQ-VAE model.

    Referred to the paper and https://github.com/MishaLaskin/vqvae/blob/master/models/vqvae.py. 
    """
    def __init__(self, n_embeddings, embeddings_dim, beta):
        super(VectorQuantizer, self).__init__()
        # Number of vectors in the codebook
        self.n_embeddings = n_embeddings
        # Dimension of embeddings - embeddings vector is of size: n_embeddings x embeddings_dim
        self.embeddings_dim = embeddings_dim
        # commitment cost - how much do we want to push the encoder output towards the closest embedding vector.
        self.beta = beta
        # Embedding object, and initialized uniformly. 
        self.embedding = nn.Embedding(self.n_embeddings, self.embeddings_dim)
        self.embedding.weight.data.uniform_(-1/self.n_embeddings, 1/self.n_embeddings)
    
    def forward(self, z):
        """
        Takes the output of the Encoder network z and maps it to a discrete
        vector (that represents the closest embedding vector e_j).
        """
        # Reshape z from BCHW to BHWC
        z = z.permute(0, 2, 3, 1).contiguous()

        # Flatten the tensor from [B,H,W,C] to [B*H*W,C].
        # This gives the number of vectors of size C, all of which will 
        # be mapped onto the codebook.
        # The channels (64) are used as the space in which to quantize. 
        z_flattened = z.view(-1, self.embeddings_dim)

        # Calculate distance from z (encoder output) to embeddings, 
        # using the formula: distance = ||z||^2 + ||e_j||^2 - 2*z*e_j. 
        # we have embedding input of (K, 64), we'll get a table that's (N, K). 
        # For every encoded vector, will find the distance to every embedding vector.
        distance = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
                   torch.sum(self.embedding.weight**2, dim=1) - \
                   2 * torch.matmul(z_flattened, self.embedding.weight.t())

        # Find the codebook vector that's the closest (Euclid) to the encoded vector. 
        # Once we find the min index, that's our approximate posterior, it's deterministic distribution (one-hot) of the input variable. 
        train_indices_return = torch.argmin(distance, dim=1)
        min_encoding_indices = torch.argmin(distance, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_embeddings).to(device)
        # use the k (min embeddings vector) as a mask, to create N x K matrix structure, 
        # where the min embedding indices are 1 in each row, and all others are 0. 
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # Get the quantized latent vectors
        # z_q(x) = e_k, where k = argmin ||z(x) - e_j||^2 as obtained above. 
        # Multiply to extract the quantized vector, we end up with N x 64 dimensional table that contains the quantized vectors. 
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        
        # During forward computation, the nearest embedding z_q is passed to the decoder, 
        # and during the backwards pass the gradient is passed unaltered to the encoder. 
        # The gradient can push the encoder's output to be discretised differently in the
        # next forward pass. 
        # To learn the embedding space, use the dictionary learning algorithm, Vector Quantisation (VQ). 
        # This uses l_2 error to move the embedding vectors e_j towards the encoder output z_e(x).
        # detatch = stop gradient. 
        embedding_loss = F.mse_loss(z_q.detach(), z)
        quantized_loss = F.mse_loss(z_q, z.detach())
        total_loss = quantized_loss + self.beta * embedding_loss

        # Copy gradients from the output z_e in decoder, to the encoder 
        # for backpropagation. The encoder and decoder share the same dimensional 
        # space, so we can easily copy the gradients. 
        z_q = z + (z_q - z).detach() 

        # Reshape z_q back to BHWC
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return embedding_loss, z_q, train_indices_return


class VQVAE(nn.Module):
    """
    VQ-VAE model for encoding and decoding images of the OASIS dataset.

    The model consists of an encoder, a vector quantizer, and a decoder, and 
    the input is passed through each one sequentially.
    
    """
    def __init__(self, n_hidden_layers, n_residual_hidden_layers, n_embeddings, embeddings_dim, beta):
        super(VQVAE, self).__init__()
        
        self.encoder = Encoder(n_inputs=1, 
                               n_hidden_layers=n_hidden_layers, 
                               n_residual_hidden_layers=n_residual_hidden_layers)
        self.conv1 = nn.Conv2d(
            in_channels=n_hidden_layers,
            out_channels=embeddings_dim,
            kernel_size=1, 
            stride=1, 
            padding=0)

        self.vector_quantizer = VectorQuantizer(
            n_embeddings=n_embeddings, 
            embeddings_dim=embeddings_dim, 
            beta=beta)
        
        self.decoder = Decoder(
            n_inputs=embeddings_dim, # 64
            n_hidden_layers=n_hidden_layers, # 64
            n_residual_hidden_layers=n_residual_hidden_layers) #32
    
    def forward(self, z):
        z = self.encoder(z)
        z = self.conv1(z)
        embedding_loss, z_q, encodings = self.vector_quantizer(z)
        reconstructed_x = self.decoder(z_q)
        return embedding_loss, reconstructed_x, z_q, encodings

class Discriminator(nn.Module):
    """
    Detect fake images from real images (decoder).

    Takes an image, and outputs a single classification representing the 
    probability of it being real or fake. 
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        # Layer size values are hard-coded here to customise the problem of 
        # generating latent tensors.
        self.net = nn.Sequential(
            # nn.Conv2d(1, 128, kernel_size=4, stride=2, padding=1), # With a stride of 2, need padding of 1 - prevents downsampling.
            nn.LeakyReLU(0.2), # Allows 0.2 of the negative
            self._block(64, 64), # map from the embeddings dimension to the image dimension
            self._block(64, 128),
            self._block(128, 256), 
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Flatten(),
            nn.Sigmoid() # convert to probability output [0,1]
        )

    # Create a discriminator block with a convolutional layer, batch normalization, and leaky ReLU activation.
    def _block(self, in_channels, out_channels):
        # Use of strided convolution is better than downsampling - model pools itself. 
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.net(x)
    
# Create the generator model.
class Generator(nn.Module):
    """ 
    Produce fake images sampled from the latent space (encoder).

    Needs to receive latent space vector as an input, and map to a vector quantized space. 
    Batch norm after the conv-transpose layers helps with vanishing gradient problem.
    """
    def __init__(self):
        super(Generator, self).__init__()
        # Input: N x channels_noise x 1 x 1
        self.net = nn.Sequential(
            self._block(128, 512, 1, 0), # go from latent space to quantized space
            self._block(512, 256, 2, 1),
            self._block(256, 128, 2, 1), 
            self._block(128, 64, 2, 1), 
            # self._block(128, 3, 2, 1), 
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False), # N x channels_img x 64 x 64
            nn.Tanh() # convert to [-1, 1] 
        )

    # Create a generator block with a transposed convolutional layer, batch normalization, and ReLU activation.
    def _block(self, in_channels, out_channels, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.net(x)

