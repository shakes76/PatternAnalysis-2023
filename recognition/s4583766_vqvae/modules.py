"""
Core components of the model required for the pattern recognition task.

Sophie Bates, s4583766.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: extract residual_block to separate module
# TODO: separate encoder and decoder into separate modules
# TODO: check and fix the layer sizes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    """
    Encoder module for the VQ-VAE model.

    The encoder consists of 2 strided convolutional layers with stride 2 and 
    window size 4 × 4, followed by two residual 3 × 3 blocks (implemented as 
    ReLU, 3x3 conv, ReLU, 1x1 conv), all having 256 hidden units.
    """

    def __init__(self, n_inputs, n_hidden_layers,n_residual_hidden_layers):
        super(Encoder, self).__init__()
        self.n_channels = n_inputs
        # self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(
            in_channels=n_inputs,
            out_channels=n_hidden_layers//2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=n_hidden_layers//2, 
            out_channels=n_hidden_layers, 
            kernel_size=4, 
            stride=2, 
            padding=1
        )

        self.residual_block_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=n_hidden_layers, 
                      out_channels=n_residual_hidden_layers, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_residual_hidden_layers, 
                      out_channels=n_hidden_layers, 
                      kernel_size=1, 
                      stride=1, 
                      padding=0),
        )
        self.residual_block_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=n_hidden_layers, 
                      out_channels=n_residual_hidden_layers, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_residual_hidden_layers, 
                      out_channels=n_hidden_layers, 
                      kernel_size=1, 
                      stride=1, 
                      padding=0),
        )
        # self.conv3 = nn.Conv2d(
        #     in_channels=n_hidden_layers, 
        #     out_channels=n_hidden_layers, 
        #     kernel_size=3, 
        #     stride=1, 
        #     padding=0
        # )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block_1(out)
        out = self.residual_block_2(out)
        out = self.relu(out)

        # out = self.conv3(out)
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
    def __init__(self, n_inputs, n_hidden_layers, n_residual_hidden_layers): #TODO: remove n_residual_layers?
        super(Decoder, self).__init__()
        # self.no_channels = n_inputs
        # self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(
            in_channels=n_inputs,
            out_channels=n_hidden_layers,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.residual_block_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=n_hidden_layers, 
                      out_channels=n_residual_hidden_layers, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_residual_hidden_layers, 
                      out_channels=n_hidden_layers, 
                      kernel_size=1, 
                      stride=1, 
                      padding=0),
        )

        self.residual_block_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=n_hidden_layers, 
                      out_channels=n_residual_hidden_layers, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_residual_hidden_layers, 
                      out_channels=n_hidden_layers, 
                      kernel_size=1, 
                      stride=1, 
                      padding=0),
        )  
        self.transpose_conv1 = nn.ConvTranspose2d(
            in_channels=n_hidden_layers,
            out_channels=n_hidden_layers//2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.transpose_conv2 = nn.ConvTranspose2d(
            in_channels=n_hidden_layers//2,
            out_channels=3, #TODO: set this no. channels as a constant in driver
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.residual_block_1(out)
        out = self.residual_block_2(out)
        out = self.transpose_conv1(out)
        out = self.transpose_conv2(out)
        return out

class VQVAE(nn.Module):
    def __init__(self, n_hidden_layers, n_residual_hidden_layers, n_embeddings, embeddings_dim, beta):
        super(VQVAE, self).__init__()
        
        self.encoder = Encoder(n_inputs=3, 
                               n_hidden_layers=n_hidden_layers, 
                               n_residual_hidden_layers=n_residual_hidden_layers)
        self.conv1 = nn.Conv2d(
            in_channels=n_hidden_layers,
            out_channels=embeddings_dim,
            kernel_size=1, 
            stride=1, 
            padding=0)

        self.vector_quantizer = VectorQuantizer(n_embeddings=n_embeddings, embeddings_dim=embeddings_dim, beta=beta)
        self.decoder = Decoder(n_inputs=embeddings_dim, 
                               n_hidden_layers=n_hidden_layers, 
                               n_residual_hidden_layers=n_residual_hidden_layers)
        # self.embedding = nn.Embedding(num_embeddings, latent_dim)
    
    def forward(self, x):
        z = self.encoder(x)
        z = self.conv1(z)
        embedding_loss, z_q = self.vector_quantizer(z)
        reconstructed_x = self.decoder(z_q)
        return embedding_loss, reconstructed_x

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

        BCHW - c starts at 1, but will increase 
        """
        # reshape z to (batch_size * height * width, channel) from BCHW.
        # i.e. [32, 64, 64, 64] to [32, 64, 64, 64].
        # memory locations are contiguous. 
        z = z.permute(0, 2, 3, 1).contiguous()

        # then flatten to [16*32*32, 64] == [16384, 64]. 
        # This gives 16384 vectors of size 64, all of which will be 'snapped' to the codebook independently. The channels (64) are used as the space in which to quantize. 
        z_flattened = z.view(-1, self.embeddings_dim)

        # Calculate distance from z (encoder output) to embeddings, 
        # using the formula: distance = ||z||^2 + ||e_j||^2 - 2*z*e_j. 
        # i.e. shape of input is (N, 64). 
        # we have embedding input of (K, 64), we'll get a table that's (N, K). For every encoded vector, will find the distance to every embedding vector.
        distance = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
                   torch.sum(self.embedding.weight**2, dim=1) - \
                   2 * torch.matmul(z_flattened, self.embedding.weight.t())

        # Find the closest encodings (k)
        # z_e(x) is encoded vector. e_j is codebook vector. 
        # Trying to find the codebook vector that's the closest to the encoded vector, where 'closeness' is the Euclidean distance. 
        # Once we find the min index, that's our approximate posterior, it's deterministic distribution (one-hot) of the input variable. 
        min_encoding_indices = torch.argmin(distance, dim=1).unsqueeze(1)
        # z_q(x) is just the closest vector to z_e(x). 
        # 
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_embeddings).to(device)
        # use the k (min embeddings vector) as a mask, to create N x K matrix structure, where the min embedding indices are 1 in each row, and all others are 0. 
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # Get the quantized latent vectors
        # z_q(x) = e_k, where k = argmin ||z(x) - e_j||^2 as obtained above. 
        # Multiply to extract the quantized vector, we end up with N x 64 dimensional table that contains the quantized vectors. 
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        
        # During forward computation, the nearest embedding z_q is passed to the decoder, and during the backwards pass the gradient is passed unaltered to the encoder. The gradient can push the encoder's output to be discretised differently in the next forward pass. 

        # To learn the embedding space, use the dictionary learning algorithm, Vector Quantisation (VQ). This uses l_2 error to move the embedding vectors e_j towards the encoder output z_e(x).
        # detatch = stop gradient. 
        # Based on the paper:
        embedding_loss = F.mse_loss(z_q.detach(), z)
        quantized_loss = F.mse_loss(z_q, z.detach())
        total_loss = quantized_loss + self.beta * embedding_loss

        # copy gradients from the output z_e in decoder, to the encoder for backpropagation. The encoder and decoder share the same dimensional space, so we can easily copy the gradients. 
        # Implementation of the straight through gradient. 
        # 
        z_q = z + (z_q - z).detach()
        # can't minimize KL in this situation because K is constant. 

        # reshape z_q back to (batch_size, height, width, channel)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return embedding_loss, z_q