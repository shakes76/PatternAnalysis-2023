"""
Hugo Burton
s4698512
20/09/2023

modules.py
Contains the source code of the components of my model
Each component will be designated as a class
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class ResBlock(nn.Module):
    """
    Defines residual block layer
    """

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class VQEmbedding(nn.Module):
    """
    Codebook in the Latent Space

    This class represents the codebook used in the Vector Quantization process within the VQ-VAE.

    Parameters:
    - K: Number of embeddings (code vectors) in the codebook.
    - D: Dimensionality of each code vector.

    The codebook is initialized with random values between -1/K and 1/K.
    """

    def __init__(self, K: int, D: int) -> None:
        super().__init__()
        # Create an embedding layer with K embeddings, each of dimension D
        self.embedding = nn.Embedding(K, D)
        # Initialize the embedding weights with values between -1/K and 1/K
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        """
        Forward pass of the codebook.

        Given the input tensor z_e_x (latent space representation), this function
        quantises it using the codebook and returns the quantised representation (latents).

        Parameters:
        - z_e_x: Latent space representation of the input.

        Returns:
        - latents: Quantised representation of the input using the codebook.
        """
        # Permute the dimensions for compatibility with the VectorQuantization function
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        # Use the custom VectorQuantization function to quantise the input
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        """
        Straight-through version of the codebook.

        This function quantises the input tensor z_e_x using the codebook and returns
        both the quantised representation (z_q_x) and the "straight-through" version
        of the quantised representation (z_q_x_bar).

        Parameters:
        - z_e_x: Latent space representation of the input.

        Returns:
        - z_q_x: Quantised representation of the input using the codebook.
        - z_q_x_bar: "Straight-through" version of the quantised representation.
        """
        # Permute the dimensions for compatibility with the VectorQuantization function
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        # Use the custom VectorQuantizationStraightThrough function to quantise the input
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        # Permute the dimensions back to the original format
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        # Retrieve the corresponding code vectors from the codebook
        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
                                               dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar


def weights_init(m: nn.Module) -> None:
    """
    Initialise the weights of convolutional layers in a neural network using Xavier initialization.

    Args:
        m (nn.Module): A layer within the neural network.

    Note:
        Xavier initialisation sets the initial weights of convolutional layers to be drawn from a 
        uniform distribution with a specific scale set by m.weight.data, and biases are 
        initialised to zeros.

    Returns:
        None
    """
    # Get the class name of the module 'm'
    classname = m.__class__.__name__

    # Check if the class name contains 'Conv' (indicating it's a convolutional layer)
    if classname.find('Conv') != -1:
        try:
            # Apply Xavier uniform initialization to the weights
            nn.init.xavier_uniform_(m.weight.data)

            # Initialize biases to zeros
            m.bias.data.fill_(0)
        except AttributeError:
            # If the layer does not have 'weight' or 'bias' attributes, skip initialization
            print("Skipping initialization of ", classname)


class VectorQuantisedVAE(nn.Module):
    """
    Vector Quantized Variational Autoencoder (VQ-VAE) Model

    This class defines a VQ-VAE model, which is used for efficient representation learning
    and generative tasks. The VQ-VAE takes an input image and quantizes/discretizes it
    into a latent space. The quantized representation is then decoded to reconstruct
    the original image or generate variations.

    Parameters:
    - input_channels: Number of channels in the input image (e.g., 1 for grayscale, 3 for RGB).
    - output_channels: Number of channels in the output image (typically 1 for grayscale).
    - num_embeddings: Number of embedded vectors in the codebook.
    - hidden_channels: Number of neurons in each hidden layer of the neural network.
    - z_dim: Dimensionality of the latent space.

    The model architecture consists of an encoder, a codebook, and a decoder.

    - Encoder: Converts input images into latent codes.
    - Codebook: Discretizes the latent codes and provides embedding vectors.
    - Decoder: Reconstructs or generates images from the discretized latent codes.

    The forward pass combines the encoder and decoder to produce the output.

    Note: The model includes codebook embeddings and custom functions for quantization.

    """

    def __init__(self, input_channels: int, output_channels: int, num_embeddings: int, hidden_channels: int) -> None:
        # Call parent constructor
        super().__init__()

        self.encoder = nn.Sequential(
            # Downconvolution (downscaling)
            nn.Conv2d(input_channels, hidden_channels, 4, 2, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            ##
            # Downconvolution (downscaling)
            nn.Conv2d(hidden_channels, hidden_channels, 4, 2, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            ##
            # Downconvolution (downscaling)
            nn.Conv2d(hidden_channels, hidden_channels, 4, 2, 1),
            nn.BatchNorm2d(hidden_channels),
            ResBlock(hidden_channels),
            ResBlock(hidden_channels),
        )

        self.codebook = VQEmbedding(num_embeddings, hidden_channels)

        self.decoder = nn.Sequential(
            # Upconvolution (upscaling)
            ResBlock(hidden_channels),
            ResBlock(hidden_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, 2, 1),
            nn.BatchNorm2d(hidden_channels),
            ##
            # Upconvolution (upscaling)
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, 2, 1),
            nn.BatchNorm2d(hidden_channels),
            ##
            # Upconvolution (upscaling)
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_channels, output_channels, 4, 2, 1),
            nn.Tanh()
        )

        # Initialize model weights
        self.apply(weights_init)

    def encode(self, x):
        """
        Encoder Network

        Given an input image 'x', this method encodes it into latent codes.

        Parameters:
        - x: Input image tensor.

        Returns:
        - latents: Encoded latent codes.
        """

        # Forward pass through the encoder
        z_e_x = self.encoder(x)
        # Quantize the latent codes using the codebook
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, z):
        """
        Decoder Network

        Given latent codes 'z', this method decodes them to reconstruct the original image.

        Parameters:
        - z: Latent code tensor.

        Returns:
        - x_tilde: Reconstructed image tensor.
        """

        # Decode the latent codes and generate the reconstructed image
        z_q_x = self.codebook.embedding(
            z).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        """
        Forward Pass

        Combines the encoder and decoder in the forward pass to produce the output.

        Parameters:
        - x: Input image tensor.

        Returns:
        - x_tilde: Reconstructed or generated image tensor.
        - z_e_x: Encoded latent codes.
        - z_q_x: Discretized latent codes.
        """

        # Forward pass through the encoder
        z_e_x = self.encoder(x)
        # Perform vector quantization using the codebook
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        # Decode the quantized latent codes to generate the output image
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x

    def forward_peturb(self, x, noise_scale: int = 0.1):
        """
        Forward Pass

        Combines the encoder and decoder in the forward pass to produce the output.
        Adds peturbance / noise to the codebook in order to generate a new / different 
        sample

        Parameters:
        - x: Input image tensor.

        Returns:
        - x_tilde: Reconstructed or generated image tensor.
        - z_e_x: Encoded latent codes.
        - z_q_x: Discretized latent codes.
        """

        # Forward pass through the encoder
        z_e_x = self.encoder(x)
        # Perform vector quantization using the codebook
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)

        # Generate random noise
        rand_noise = torch.randn_like(z_q_x_st) * noise_scale

        # Add noise to z_q_x_st
        z_q_x_st_noisy = z_q_x_st + rand_noise

        # Decode the now noisy quantised latent codes to generate the output image
        # the output image will now be a perturbed version of the input
        x_tilde = self.decoder(z_q_x_st_noisy)

        return x_tilde, z_e_x, z_q_x


class VectorQuantisation(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                                    inputs_flatten, codebook.t(), alpha=-2.0, beta=0.3)

            _, indices_flatten = torch.min(distances, dim=1)

            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices


class VectorQuantisationStraightThrough(Function):
    """
    This class defines a custom autograd function used in the training of Variational Autoencoders 
    with Vector Quantisation (VQ-VAEs). It implements the "straight-through estimator" technique, 
    allowing for the quantisation of continuous data into discrete representations while enabling 
    gradient backpropagation during training.

    Since we are discretising the latent space, it is no longer to compute gradients. Therefore, 
    instead we use a trick by copying the gradients from the decoder "straight-through" back into 
    the encoder for backpropagation.
    """

    @staticmethod
    def forward(ctx, inputs, codebook):
        # Step 1: Quantisation
        # Quantise/discretise the input using the codebook
        indices = vq(inputs, codebook)
        # Flatten the indices for further processing
        indices_flatten = indices.view(-1)

        # Save relevant data for the backward pass
        ctx.save_for_backward(indices_flatten, codebook)

        # Mark indices_flatten as non-differentiable since it's an index
        ctx.mark_non_differentiable(indices_flatten)

        # Retrieve the code vectors corresponding to the quantised indices
        codes_flatten = torch.index_select(codebook, dim=0,
                                           index=indices_flatten)
        # Reshape the code vectors to match the shape of the input
        codes = codes_flatten.view_as(inputs)

        # Return both the quantised codes and the flattened indices
        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, _):
        # Initialise gradient variables
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Step 2: Backpropagation through quantization
            # The straight-through estimator assigns the gradient of the output
            # to the input without modification
            grad_inputs = grad_output.clone()

        if ctx.needs_input_grad[1]:
            # Step 3: Backpropagation through codebook
            # Gradient computation with respect to the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            # Flatten the gradient
            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            # Initialize the gradient for the codebook
            grad_codebook = torch.zeros_like(codebook)
            # Accumulate the gradients for each code vector based on indices
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        # Return the computed gradients for inputs and codebook
        return (grad_inputs, grad_codebook)


# Alias for the custom functions, which can be used later
vq = VectorQuantisation.apply
vq_st = VectorQuantisationStraightThrough.apply
__all__ = [vq, vq_st]

# test case
if __name__ == "__main__":
    batch_size = 4
    img_x, img_y = 256, 256

    x = torch.rand(batch_size, img_x * img_y)
    vae = VectorQuantisedVAE(input_channels=img_x * img_y,
                             output_channels=img_x * img_y, num_embeddings=128, hidden_channels=32)
    x_recon, mu, sigma = vae(x)
    print(x_recon.shape, mu.shape, sigma.shape)
