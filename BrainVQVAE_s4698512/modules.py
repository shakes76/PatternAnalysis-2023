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


class VectorQuantizedVAE(nn.Module):
    """
    Takes a tensor and quantises/discretises it
    Input image > Hidden Dimension -> Output mean and standard deviation with parametrisation trick
    > Take that to the decoder > Output image
    """

    def __init__(self, input_dim, hidden_dim=200, z_dim=20, K=512):
        # Call parent
        super().__init__()

        # Encoder
        self.img_to_hidden = nn.Linear(input_dim, hidden_dim)

        # Parametrisation
        self.hidden_to_mu = nn.Linear(hidden_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(hidden_dim, z_dim)

        # Decoder (pretty much opposite of encoder)
        self.z_to_hidden = nn.Linear(z_dim, hidden_dim)
        self.hidden_to_img = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(input_dim, dim, 4, 2, 1),
        #     nn.BatchNorm2d(dim),
        #     nn.ReLU(True),
        #     nn.Conv2d(dim, dim, 4, 2, 1),
        #     ResBlock(dim),
        #     ResBlock(dim),
        # )

        self.codebook = VQEmbedding(K, dim)

        # self.decoder = nn.Sequential(
        #     ResBlock(dim),
        #     ResBlock(dim),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(dim, dim, 4, 2, 1),
        #     nn.BatchNorm2d(dim),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
        #     nn.Tanh()
        # )

        # self.apply(weights_init)

    def encode(self, x):
        """
        q_phi(z|x)
        Encoder network
        """

        h = self.relu(self.img_to_hidden(x))

        mu, sigma = self.hidden_to_mu(h), self.hidden_to_sigma(h)
        return mu, sigma

        # z_e_x = self.encoder(x)
        # latents = self.codebook(z_e_x)
        # return latents

    def decode(self, z):
        """
        send in the latent image and try to recover the original
        p_theta(x|z)
        """

        # take in z
        h = self.relu(self.z_to_hidden(z))

        # Ensure decode is between 0 and 1 by using sigmoid
        return torch.sigmoid(self.hidden_to_img(h))

        # z_q_x = self.codebook.embedding(
        #     latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        # x_tilde = self.decoder(z_q_x)
        # return x_tilde

    def forward(self, x):
        """
        Combines encoder and decoder in the forward pass
        """

        # Generate mu and sigma
        mu, sigma = self.encode(x)
        epsilon = torch.rand_like(sigma)
        z_reparam = mu + sigma * epsilon
        # gaussian model ^^

        # Decode
        x_recon = self.decode(z_reparam)
        return x_recon, mu, sigma

        # z_e_x = self.encoder(x)
        # z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        # x_tilde = self.decoder(z_q_x_st)
        # return x_tilde, z_e_x, z_q_x


# test case
if __name__ == "__main__":
    batch_size = 4
    img_x, img_y = 28, 28

    x = torch.rand(batch_size, img_x * img_y)
    vae = VectorQuantizedVAE(input_dim=img_x * img_y)
    x_recon, mu, sigma = vae(x)
    print(x_recon.shape, mu.shape, sigma.shape)
