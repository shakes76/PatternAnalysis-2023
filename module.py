import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


class ResnetBlock(nn.Module):
    '''
        Resblock:
                      (w/o Dropout)    (with Dropout)
            Input ->    Conv2d ------------> Conv2d ------> Output
                \\       Time Embedding /            /
                 \\--(shortcut if size not fit)------
    '''

    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, time_emb_size=None):
        super().__init__()

        # Record useful arguments
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        # Define Resblocks 1
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        )

        # Define Time Embedding
        if time_emb_size:
            self.time_emb = nn.Sequential(
                nn.Linear(time_emb_size, out_channels),
                nn.SiLU(),
            )

        # Define Resblocks 2
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

        # Define shortcut
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x, time_encode=None):
        h = x

        # Pass block 1
        h = self.block1(h)
        # Add time embedding / encoding
        if time_encode is not None:
            if self.time_emb is None:
                raise ValueError("This resblock has no time embedding.")
            h = h + self.time_emb(time_encode)[:, :, None, None]
        # Pass block 2
        h = self.block2(h)
        # If size not match, add shortcut to make shortcut(x).shape = resblock(x).shape.
        if self.in_channels != self.out_channels:
            x = self.shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    '''
        Do Self-attention & residual.
    '''

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        # Input normalization
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels)

        # Crucial weights of three vectors
        self.q = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.k = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.v = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

        # Projection after attention
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

    def forward(self, X):

        # Copy vector for skip connection
        H = X
        b, c, h, w = X.shape

        # Do normalization before self-attention
        H = self.norm(H)

        # Compute three crucial vectors.
        Q = self.q(H)
        K = self.k(H)
        V = self.v(H)

        # Reshape matrix for calculate W=QK
        Q = rearrange(Q, "b c h w -> b (h w) c")
        K = rearrange(K, "b c h w -> b c (h w)")
        W = torch.bmm(Q, K)
        # w.shape=(b,hw,hw),  w[b,i,j] = sum_c q[b,i,c]k[b,c,j], j->key, i->it's value pair

        # Normalize the score
        W = nn.functional.softmax(W * (int(c)**(-0.5)), dim=2)

        # attend to values
        V = rearrange(V, "b c h w -> b c (h w)")
        W = rearrange(W, "b hw_q hw_k -> b hw_k hw_q")
        H = torch.bmm(V, W)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] W[b,i,j], j->key

        # rearrange output and projection
        H = rearrange(H, "b c (h w) -> b c h w", h=h)
        H = self.proj_out(H)

        # Skip connection
        return X+H


class Downsample(nn.Module):
    '''
        Do downsample. Shape should change from [2n, 2n] -> [n, n]
    '''

    def __init__(self, in_channels, with_conv):
        super().__init__()
        if with_conv:
            # Note that original papper do:
            # 1. pad (0, 1, 0, 1) and go through nn.Conv2d(inc, inc, 3, 2, 0)
            # 2. nn.Conv2d(inc, inc, 3, 2, 0)

            # Here we use nn.Conv2d(inc, inc, 3, 2, 1) to instead.
            # You can see the diffenece between these two in playground/conv_padding
            self.layer = torch.nn.Conv2d(in_channels, in_channels, 3, 2, 1)
        else:
            self.layer = torch.nn.AvgPool2d(2)

    def forward(self, x):
        return self.layer(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        # Just do interpolate & convolution
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class VectorQuantizer2(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.re_embed = n_e

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        return z_q, loss, min_encoding_indices