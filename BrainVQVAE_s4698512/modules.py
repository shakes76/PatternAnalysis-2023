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
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
import torchvision.transforms as transforms


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
    """

    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
                                               dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class VectorQuantizedVAE(nn.Module):
    """
    Takes a tensor and quantises/discretises it
    Input image > Hidden Dimension -> Output mean and standard deviation with parametrisation trick
    > Take that to the decoder > Output image

    Input channels is the number of channels an input image has. E.g. greyscale = 1, RGB = 3
    Output channels is the number of channels the output image from the model has. For my problem, will be 1
    Hidden channels is essentially the number of neurons in each layer (that isn't the input or output) within the neural network
    They are called hidden layers because you don't get to see them. They are essentially between the input and the output.

    NumEmbeddings is the number of embedded vectors in the codebook
    """

    def __init__(self, input_channels, output_channels, num_embeddings=512, hidden_channels=200, z_dim=20):
        # Call parent
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, num_embeddings,
                      kernel_size=1)  # Output latent codes
        )

        self.codebook = VQEmbedding(num_embeddings, hidden_channels)

        self.decoder = nn.Sequential(
            nn.Conv2d(num_embeddings, hidden_channels, kernel_size=1),
            nn.ConvTranspose2d(hidden_channels, hidden_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, output_channels, kernel_size=1),
            nn.Sigmoid()  # Assuming output is in [0, 1] range (for grayscale)
        )

        self.apply(weights_init)

    def encode(self, x):
        """
        q_phi(z|x)
        Encoder network
        """
        print("encode")

        # h = self.relu(self.img_to_hidden(x))

        # mu, sigma = self.hidden_to_mu(h), self.hidden_to_sigma(h)
        # return mu, sigma

        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, z):
        """
        send in the latent image and try to recover the original
        p_theta(x|z)
        """
        print("decode")

        # take in z
        # h = self.relu(self.z_to_hidden(z))

        # Ensure decode is between 0 and 1 by using sigmoid
        # return torch.sigmoid(self.hidden_to_img(h))

        z_q_x = self.codebook.embedding(
            z).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        """
        Combines encoder and decoder in the forward pass
        """

        # Generate mu and sigma
        # mu, sigma = self.encode(x)
        # epsilon = torch.rand_like(sigma)
        # z_reparam = mu + sigma * epsilon
        # # gaussian model ^^

        # # Decode
        # x_recon = self.decode(z_reparam)
        # return x_recon, mu, sigma

        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x


class VectorQuantization(Function):
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
                                    inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
                           '`VectorQuantization`. The function `VectorQuantization` '
                           'is not differentiable. Use `VectorQuantizationStraightThrough` '
                           'if you want a straight-through estimator of the gradient.')


class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
                                           index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)


vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply
__all__ = [vq, vq_st]

# test case
if __name__ == "__main__":
    batch_size = 4
    img_x, img_y = 28, 28

    x = torch.rand(batch_size, img_x * img_y)
    vae = VectorQuantizedVAE(input_channels=img_x * img_y)
    x_recon, mu, sigma = vae(x)
    print(x_recon.shape, mu.shape, sigma.shape)
