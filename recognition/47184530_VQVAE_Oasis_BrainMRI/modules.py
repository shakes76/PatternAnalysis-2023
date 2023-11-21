import torch
import torch.nn.functional as F
from torch import nn

# The Vector Quantizer layer performs the quantization of the encoder's outputs.
# This is where the continuous representations from the encoder are mapped to a discrete set of embeddings.
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super(VectorQuantizer, self).__init__()

        # Embedding dimension: size of each embedding vector
        self.embedding_dim = embedding_dim

        # Number of embeddings: total number of discrete embeddings in our codebook
        self.num_embeddings = num_embeddings

        # Beta is a hyperparameter that weights the commitment loss
        self.beta = beta

        # Initialize the embeddings (codebook) with random values. It's a learnable parameter.
        self.embeddings = nn.Parameter(torch.randn(embedding_dim, num_embeddings))

    def forward(self, x):
        # Reshape the tensor to compute distances
        z_e_x = x.permute(0, 2, 3, 1).contiguous()
        z_e_x_ = z_e_x.view(-1, self.embedding_dim)

        # Compute pairwise distances between input and the codebook
        distances = (torch.sum(z_e_x_**2, dim=1, keepdim=True)
                    + torch.sum(self.embeddings**2, dim=0)
                    - 2 * torch.matmul(z_e_x_, self.embeddings))

        # Find the closest embedding index for each item in the batch
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # Create a one-hot encoding of the indices
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings).to(x.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Reshape the encoding indices to have the same spatial dimensions as input
        encoding_indices = encoding_indices.view(*z_e_x.shape[:-1])

        # Use the encodings to get the quantized values from the codebook
        quantized = torch.matmul(encodings, self.embeddings.t()).view(*z_e_x.shape)

        # Compute the commitment loss and the quantization loss
        e_latent_loss = F.mse_loss(quantized.detach(), z_e_x)
        q_latent_loss = F.mse_loss(quantized, z_e_x.detach())
        loss = q_latent_loss + self.beta * e_latent_loss

        # Straight-through estimator: gradients bypass the non-differentiable operation
        quantized = z_e_x + (quantized - z_e_x).detach()

        # Compute perplexity to check how many codebook entries are being used
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encoding_indices

# The Encoder module maps the input images to a continuous representation that will be quantized by the Vector Quantizer.
class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, embedding_dim):
        super(Encoder, self).__init__()

        # Define the encoder neural network
        # The encoder consists of three convolutional layers with ReLU activations.
        self.encoder = nn.Sequential(
            # First convolutional layer: it takes the input image and produces 'hidden_channels' feature maps.
            nn.Conv2d(input_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            # Second convolutional layer: reduces the spatial dimensions by half and reduces the number of feature maps.
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            # Third convolutional layer: prepares the tensor for quantization by setting the number of channels to 'embedding_dim'.
            nn.Conv2d(hidden_channels // 2, embedding_dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Forward propagation of input through the encoder
        return self.encoder(x)

# The Decoder module maps the quantized representation back to the space of the original image.
class Decoder(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(Decoder, self).__init__()

        # Define the decoder neural network
        # The decoder consists of three transposed convolutional layers (sometimes called "deconvolutional layers") with ReLU activations.
        self.decoder = nn.Sequential(
            # First transposed convolutional layer: it takes the quantized representation and increases the spatial dimensions.
            nn.ConvTranspose2d(input_channels, hidden_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            # Second transposed convolutional layer: further increases the spatial dimensions.
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            # Third transposed convolutional layer: produces the final output with the same shape as the original image.
            nn.ConvTranspose2d(hidden_channels // 2, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Forward propagation of the quantized representation through the decoder
        return self.decoder(x)

# The VQ-VAE module combines the encoder, vector quantizer, and decoder components.
class VQVAE(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_embeddings, embedding_dim):
        super(VQVAE, self).__init__()

        # Initialize the encoder module
        self.encoder = Encoder(input_channels, hidden_channels, embedding_dim)

        # Initialize the vector quantization module
        self.quantize = VectorQuantizer(num_embeddings, embedding_dim)

        # Initialize the decoder module
        self.decoder = Decoder(embedding_dim, hidden_channels)

    def forward(self, x):
        # Encode the input image to a continuous representation
        z = self.encoder(x)

        # Quantize the continuous representation
        loss, quantized, perplexity, _ = self.quantize(z)

        # Decode the quantized representation to produce the reconstruction
        x_recon = self.decoder(quantized)

        return loss, x_recon, perplexity

# The VQVAETrainer module facilitates the training of the VQ-VAE model.
class VQVAETrainer(nn.Module):
    def __init__(self, train_variance, input_channels, hidden_channels, num_embeddings, embedding_dim):
        super(VQVAETrainer, self).__init__()

        # Store the variance of the training data (used for normalization)
        self.train_variance = train_variance

        # Initialize the VQ-VAE model
        self.vqvae = VQVAE(input_channels, hidden_channels, num_embeddings, embedding_dim)

    def forward(self, x):
        # Forward propagation of the input through the VQ-VAE
        vq_loss, x_recon, perplexity = self.vqvae(x)

        # Compute the reconstruction loss normalized by the training data variance
        recon_loss_value = F.mse_loss(x_recon, x) / self.train_variance

        # Overall loss is the sum of reconstruction loss and vector quantization loss
        loss = recon_loss_value + vq_loss

        return x_recon, perplexity, loss

# The PixelConvLayer is a custom convolutional layer used in the PixelCNN.
# It ensures that each pixel only depends on other pixels above it or to its left.
class PixelConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mask_type, **kwargs):
        super(PixelConvLayer, self).__init__()

        # Define the mask type (either 'A' or 'B')
        self.mask_type = mask_type

        # Compute padding to ensure the convolution is 'same' (output size == input size)
        self.padding = (kernel_size - 1) // 2

        # Define the convolutional layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs, padding=self.padding)

        # Initialize the mask to be applied on the convolutional weights
        self.mask = self.conv.weight.data.clone()

        # Create the mask
        self.create_mask()

    def forward(self, x):
        # Apply the mask to the convolutional weights
        self.conv.weight.data *= self.mask.to(self.conv.weight.device)

        # Apply the convolution
        return self.conv(x)

    def create_mask(self):
        _, _, H, W = self.conv.weight.size()

        # Set the mask to ones initially
        self.mask.fill_(1)

        # For mask type 'A', the center pixel and all pixels to the right are set to zero
        # For mask type 'B', all pixels to the right of the center pixel are set to zero
        self.mask[:, :, H // 2, W // 2 + (self.mask_type == 'A'):] = 0

        # All pixels below the center pixel are set to zero
        self.mask[:, :, H // 2 + 1:] = 0

# The PixelCNN model comprises several PixelConvLayers.
class PixelCNN(nn.Module):
    def __init__(self, input_shape, num_embeddings, embedding_dim):
        super(PixelCNN, self).__init__()

        # Define the input shape of the image
        self.input_shape = input_shape

        # Define the embedding dimension
        self.embedding_dim = embedding_dim

        # Define the number of embeddings (or the number of different pixel values)
        self.num_embeddings = num_embeddings

        # Define the architecture of the PixelCNN
        self.layers = nn.ModuleList()

        # The first layer has a mask type 'A'
        self.layers.append(PixelConvLayer(input_shape[0], embedding_dim, 7, mask_type='A'))

        # Subsequent layers have a mask type 'B'
        for _ in range(5):
            self.layers.append(PixelConvLayer(embedding_dim, embedding_dim, 7, mask_type='B'))

        # The final layer reduces the number of channels to the number of embeddings
        self.layers.append(nn.Conv2d(embedding_dim, num_embeddings, 1))

    def forward(self, x):
        # Forward propagation through the PixelCNN
        for layer in self.layers:
            x = F.relu(layer(x))
        return x
