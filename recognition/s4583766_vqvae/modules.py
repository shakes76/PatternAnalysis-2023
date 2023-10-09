"""
Core components of the model required for the pattern recognition task.

Sophie Bates, s4583766.
"""
import torch
import torch.nn as nn

# TODO: extract residual_block to separate module
# TODO: separate encoder and decoder into separate modules
# TODO: check and fix the layer sizes

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
            out_channels=64,
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

class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck module for the VQ-VAE model.

    Referred to the paper and https://github.com/MishaLaskin/vqvae/blob/master/models/vqvae.py. 
    """
    def __init__(self, no_embeddings, embeddings_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.no_embeddings = no_embeddings
        self.embeddings_dim = embeddings_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.no_embeddings, self.embeddings_dim)
        self.embedding.weight.data.uniform_(-1/self.no_embeddings, 1/self.no_embeddings)
    
    def forward(self, z):
        """
        Takes the output of the Encoder network z and maps it to a discrete
        vector (that represents the closest embedding vector e_j).
        """
        z = z.permute(0, 2, 3, 1).contiguous()
        # reshape z to (batch_size * height * width, channel)
        z_flattened = z.view(-1, self.embeddings_dim)

        # Calculate distance from z (encoder output) to embeddings, 
        # using the formula: distance = ||z||^2 + ||e_j||^2 - 2*z*e_j
        distance = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
                   torch.sum(self.embedding.weight**2, dim=1) - \
                   2 * torch.matmul(z_flattened, self.embedding.weight.t())

        # Find the closest encodings (k)
        min_encoding_indices = torch.argmin(distance, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.no_embeddings).to(device)
        # use the k (min embeddings vector) as a mask
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # Get the quantized latent vectors
        # z_q(x) = e_k, where k = argmin ||z(x) - e_j||^2 as obtained above
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        
        # During forward computation, the nearest embedding z_q is passed to the decoder, and during the backwards pass the gradient is passed unaltered to the encoder. The gradient can push the encoder's output to be discretised differently in the next forward pass. 

        # To learn the embedding space, use the dictionary learning algorithm, Vector Quantisation (VQ). This uses l_2 error to move the embedding vectors e_j towards the encoder output z_e(x).
        # Based on the paper:
        embedding_loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)

        z_q = z + (z_q - z).detach()

        # reshape z_q to (batch_size, height, width, channel)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return embedding_loss, z_q