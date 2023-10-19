import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

# --------------------------------
# VQVAE MODEL

"""The VQ-VAE Model"""
class VQVAE(nn.Module):

    def __init__(self, num_embeddings, embedding_dim):
        super(VQVAE, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.encoder = Encoder()
        self.quantiser = Quantiser(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.decoder = Decoder()
        
    def forward(self, x):
        # Input shape is B, C, H, W
        quant_input = self.encoder(x)
        quant_out, quant_loss, encoding_indices = self.quantiser(quant_input)
        output = self.decoder(quant_out)
        
        # Reconstruction Loss, and find the total loss
        reconstruction_loss = F.mse_loss(x, output)
        total_loss = quant_loss + reconstruction_loss
        
        return output, total_loss, encoding_indices
    
    """Function while allows output to be calculated directly from indices
    param quant_out_shape is the shape that the quantiser is expected to return"""
    @torch.no_grad()
    def img_from_indices(self, indices, quant_out_shape):
        quant_out = self.quantiser.output_from_indices(indices, quant_out_shape)   # Output is currently 32*32 img with 32 channels
        return self.decoder(quant_out)

"""The Encoder Model used in VQ-VAE"""
class Encoder(nn.Module):
    def __init__(self, ):
        super(Encoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
    def forward(self, x):
        out = self.encoder(x)
        return out

"""The VectorQuantiser Model used in VQ-VAE"""
class Quantiser(nn.Module):
    def __init__(self, num_embeddings, embedding_dim) -> None:
        super(Quantiser, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = 0.2
        
        self.embedding = self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    """Returns the encoding indices from the input"""
    def get_encoding_indices(self, quant_input):
        # Flatten
        quant_input = quant_input.permute(0, 2, 3, 1)
        quant_input = quant_input.reshape((quant_input.size(0), -1, quant_input.size(-1)))
    
        # Compute pairwise distances
        dist = torch.cdist(quant_input, self.embedding.weight[None, :].repeat((quant_input.size(0), 1, 1)))
        
        # Find index of nearest embedding
        encoding_indices = torch.argmin(dist, dim=-1)       # in form B, W*H
        return encoding_indices
    
    """Returns the output from the encoding indices"""
    def output_from_indices(self, indices, output_shape):
        quant_out = torch.index_select(self.embedding.weight, 0, indices.view(-1))
        quant_out = quant_out.reshape(output_shape).permute(0, 3, 1, 2)
        return quant_out
    
    def forward(self, quant_input):
        # Finds the encoding indices
        encoding_indices = self.get_encoding_indices(quant_input)
        # Gets the output based on the encoding indices
        quant_out = self.output_from_indices(encoding_indices, quant_input.shape)
        
        # Losses
        commitment_loss = torch.mean((quant_out.detach() - quant_input)**2)
        codebook_loss = torch.mean((quant_out - quant_input.detach())**2)
        loss = codebook_loss + self.beta*commitment_loss
        
        # Straight through gradient estimator for backprop
        quant_out = quant_input + (quant_out - quant_input).detach()

        # Reshapes encoding indices to 'B, H, W'
        encoding_indices = encoding_indices.reshape((-1, quant_out.size(-2), quant_out.size(-1)))
        
        return quant_out, loss, encoding_indices

"""The Decoder Model used in VQ-VAE"""
class Decoder(nn.Module):
    def __init__(self, ) -> None:
        super(Decoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1),
        )
        
    def forward(self, x):
        out = self.decoder(x)
        return out