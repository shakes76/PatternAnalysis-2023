""" VQVAE2 Moduels """

import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from functools import partial, lru_cache

import numpy as np


def wn_linear(in_dim, out_dim):
    return nn.utils.weight_norm(nn.Linear(in_dim, out_dim))

class WNConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        activation=None,
    ):
        super().__init__()

        self.conv = nn.utils.weight_norm(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        )

        self.out_channel = out_channel

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]

        self.kernel_size = kernel_size

        self.activation = activation

    def forward(self, input):
        out = self.conv(input)

        if self.activation is not None:
            out = self.activation(out)

        return out


def shift_down(input, size=1):
    return F.pad(input, [0, 0, size, 0])[:, :, : input.shape[2], :]


def shift_right(input, size=1):
    return F.pad(input, [size, 0, 0, 0])[:, :, :, : input.shape[3]]


class CausalConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding='downright',
        activation=None,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 2

        self.kernel_size = kernel_size

        if padding == 'downright':
            pad = [kernel_size[1] - 1, 0, kernel_size[0] - 1, 0]

        elif padding == 'down' or padding == 'causal':
            pad = kernel_size[1] // 2

            pad = [pad, pad, kernel_size[0] - 1, 0]

        self.causal = 0
        if padding == 'causal':
            self.causal = kernel_size[1] // 2

        self.pad = nn.ZeroPad2d(pad)

        self.conv = WNConv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
            padding=0,
            activation=activation,
        )

    def forward(self, input):
        out = self.pad(input)

        if self.causal > 0:
            self.conv.conv.weight_v.data[:, :, -1, self.causal :].zero_()

        out = self.conv(out)

        return out


class GatedResBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        channel,
        kernel_size,
        conv='wnconv2d',
        activation=nn.ELU,
        dropout=0.1,
        auxiliary_channel=0,
        condition_dim=0,
    ):
        super().__init__()

        if conv == 'wnconv2d':
            conv_module = partial(WNConv2d, padding=kernel_size // 2)

        elif conv == 'causal_downright':
            conv_module = partial(CausalConv2d, padding='downright')

        elif conv == 'causal':
            conv_module = partial(CausalConv2d, padding='causal')

        self.activation = activation()
        self.conv1 = conv_module(in_channel, channel, kernel_size)

        if auxiliary_channel > 0:
            self.aux_conv = WNConv2d(auxiliary_channel, channel, 1)

        self.dropout = nn.Dropout(dropout)

        self.conv2 = conv_module(channel, in_channel * 2, kernel_size)

        if condition_dim > 0:
            # self.condition = nn.Linear(condition_dim, in_channel * 2, bias=False)
            self.condition = WNConv2d(condition_dim, in_channel * 2, 1, bias=False)

        self.gate = nn.GLU(1)

    def forward(self, input, aux_input=None, condition=None):
        out = self.conv1(self.activation(input))

        if aux_input is not None:
            out = out + self.aux_conv(self.activation(aux_input))

        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)

        if condition is not None:
            condition = self.condition(condition)
            out += condition
            # out = out + condition.view(condition.shape[0], 1, 1, condition.shape[1])

        out = self.gate(out)
        out += input

        return out


@lru_cache(maxsize=64)
def causal_mask(size):
    shape = [size, size]
    mask = np.triu(np.ones(shape), k=1).astype(np.uint8).T
    start_mask = np.ones(size).astype(np.float32)
    start_mask[0] = 0

    return (
        torch.from_numpy(mask).unsqueeze(0),
        torch.from_numpy(start_mask).unsqueeze(1),
    )


class CausalAttention(nn.Module):
    def __init__(self, query_channel, key_channel, channel, n_head=8, dropout=0.1):
        super().__init__()

        self.query = wn_linear(query_channel, channel)
        self.key = wn_linear(key_channel, channel)
        self.value = wn_linear(key_channel, channel)

        self.dim_head = channel // n_head
        self.n_head = n_head

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key):
        batch, _, height, width = key.shape

        def reshape(input):
            return input.view(batch, -1, self.n_head, self.dim_head).transpose(1, 2)

        query_flat = query.view(batch, query.shape[1], -1).transpose(1, 2)
        key_flat = key.view(batch, key.shape[1], -1).transpose(1, 2)
        query = reshape(self.query(query_flat))
        key = reshape(self.key(key_flat)).transpose(2, 3)
        value = reshape(self.value(key_flat))

        attn = torch.matmul(query, key) / sqrt(self.dim_head)
        mask, start_mask = causal_mask(height * width)
        mask = mask.type_as(query)
        start_mask = start_mask.type_as(query)
        attn = attn.masked_fill(mask == 0, -1e4)
        attn = torch.softmax(attn, 3) * start_mask
        attn = self.dropout(attn)

        out = attn @ value
        out = out.transpose(1, 2).reshape(
            batch, height, width, self.dim_head * self.n_head
        )
        out = out.permute(0, 3, 1, 2)

        return out


class PixelBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        channel,
        kernel_size,
        n_res_block,
        attention=True,
        dropout=0.1,
        condition_dim=0,
    ):
        super().__init__()

        resblocks = []
        for i in range(n_res_block):
            resblocks.append(
                GatedResBlock(
                    in_channel,
                    channel,
                    kernel_size,
                    conv='causal',
                    dropout=dropout,
                    condition_dim=condition_dim,
                )
            )

        self.resblocks = nn.ModuleList(resblocks)

        self.attention = attention

        if attention:
            self.key_resblock = GatedResBlock(
                in_channel * 2 + 2, in_channel, 1, dropout=dropout
            )
            self.query_resblock = GatedResBlock(
                in_channel + 2, in_channel, 1, dropout=dropout
            )

            self.causal_attention = CausalAttention(
                in_channel + 2, in_channel * 2 + 2, in_channel // 2, dropout=dropout
            )

            self.out_resblock = GatedResBlock(
                in_channel,
                in_channel,
                1,
                auxiliary_channel=in_channel // 2,
                dropout=dropout,
            )

        else:
            self.out = WNConv2d(in_channel + 2, in_channel, 1)

    def forward(self, input, background, condition=None):
        out = input

        for resblock in self.resblocks:
            out = resblock(out, condition=condition)

        if self.attention:
            key_cat = torch.cat([input, out, background], 1)
            key = self.key_resblock(key_cat)
            query_cat = torch.cat([out, background], 1)
            query = self.query_resblock(query_cat)
            attn_out = self.causal_attention(query, key)
            out = self.out_resblock(out, attn_out)

        else:
            bg_cat = torch.cat([out, background], 1)
            out = self.out(bg_cat)

        return out


class CondResNet(nn.Module):
    def __init__(self, in_channel, channel, kernel_size, n_res_block):
        super().__init__()

        blocks = [WNConv2d(in_channel, channel, kernel_size, padding=kernel_size // 2)]

        for i in range(n_res_block):
            blocks.append(GatedResBlock(channel, channel, kernel_size))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class PixelSNAIL(nn.Module):
    def __init__(
        self,
        shape,
        n_class,
        channel,
        kernel_size,
        n_block,
        n_res_block,
        res_channel,
        attention=True,
        dropout=0.1,
        n_cond_res_block=0,
        cond_res_channel=0,
        cond_res_kernel=3,
        n_out_res_block=0,
    ):
        super().__init__()

        height, width = shape

        self.n_class = n_class

        if kernel_size % 2 == 0:
            kernel = kernel_size + 1

        else:
            kernel = kernel_size

        self.horizontal = CausalConv2d(
            n_class, channel, [kernel // 2, kernel], padding='down'
        )
        self.vertical = CausalConv2d(
            n_class, channel, [(kernel + 1) // 2, kernel // 2], padding='downright'
        )

        coord_x = (torch.arange(height).float() - height / 2) / height
        coord_x = coord_x.view(1, 1, height, 1).expand(1, 1, height, width)
        coord_y = (torch.arange(width).float() - width / 2) / width
        coord_y = coord_y.view(1, 1, 1, width).expand(1, 1, height, width)
        self.register_buffer('background', torch.cat([coord_x, coord_y], 1))

        self.blocks = nn.ModuleList()

        for i in range(n_block):
            self.blocks.append(
                PixelBlock(
                    channel,
                    res_channel,
                    kernel_size,
                    n_res_block,
                    attention=attention,
                    dropout=dropout,
                    condition_dim=cond_res_channel,
                )
            )

        if n_cond_res_block > 0:
            self.cond_resnet = CondResNet(
                n_class, cond_res_channel, cond_res_kernel, n_cond_res_block
            )

        out = []

        for i in range(n_out_res_block):
            out.append(GatedResBlock(channel, res_channel, 1))

        out.extend([nn.ELU(inplace=True), WNConv2d(channel, n_class, 1)])

        self.out = nn.Sequential(*out)

    def forward(self, input, condition=None, cache=None):
        if cache is None:
            cache = {}
        batch, height, width = input.shape
        input = (
            F.one_hot(input, self.n_class).permute(0, 3, 1, 2).type_as(self.background)
        )
        horizontal = shift_down(self.horizontal(input))
        vertical = shift_right(self.vertical(input))
        out = horizontal + vertical

        background = self.background[:, :, :height, :].expand(batch, 2, height, width)

        if condition is not None:
            if 'condition' in cache:
                condition = cache['condition']
                condition = condition[:, :, :height, :]

            else:
                condition = (
                    F.one_hot(condition, self.n_class)
                    .permute(0, 3, 1, 2)
                    .type_as(self.background)
                )
                condition = self.cond_resnet(condition)
                condition = F.interpolate(condition, scale_factor=2)
                cache['condition'] = condition.detach().clone()
                condition = condition[:, :, :height, :]

        for block in self.blocks:
            out = block(out, background, condition=condition)

        out = self.out(out)

        return out, cache


class VectorQuantizer(nn.Module):
    """Vector Quantizer

    Input any tensor to be quantized. Last dimension will be used as space in
    which to quantize. All other dimensions will be flattened and will be seen
    as different examples to quantize.

    The output tensor will have the same shape as the input.
    """
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        """
        Initialize a Vector Quantizer 

        Args:
            dim: dimension of data
            n_embed: number of vectors in the quantized space
            decay: embedding table decay
            eps: constant to avoid division by zero
        """
        super().__init__()

        self.dim = dim              # dimension of data
        self.n_embed = n_embed      # number of embedding vectors
        self.decay = decay          # embedding table decay
        self.eps = eps              # constant to avoid division by zero

        embed = torch.randn(dim, n_embed)       # initialize the embedding table with random values
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        """ 
        Forward Propagation
        
        Args:
            input: Tensor of the input data, final dimension must be equal to embedding_dim
        
        Returns:
            quantize: Tensor containing the quantized version of the input
            diff: quantization loss
            encoding_indices: Tensor containing the discrete encoding indices
        """
        flatten = input.reshape(-1, self.dim)           # flatten input
        dist = (                                        # get distance between each input vector and embeding
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)                   # get the closest embeddings to each input vector
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)   # convert indices to one-hot format
        embed_ind = embed_ind.view(*input.shape[:-1])   # reshape indices to match the shape of input
        quantize = self.embed_code(embed_ind)           # get quantized version of the input

        if self.training:                               # update the embedding table during training
            embed_onehot_sum = embed_onehot.sum(0)              # count time of reference for each embedding
            embed_sum = flatten.transpose(0, 1) @ embed_onehot  # get weighted sum of input vectors

            self.cluster_size.data.mul_(self.decay).add_(       # update cluster size
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)  # update average embedding value
            n = self.cluster_size.sum()                         # number of assigned vectors
            cluster_size = (                                    # normalize the cluster size
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)   # normalized average embedding value
            self.embed.data.copy_(embed_normalized)             # update the embedding table

        diff = (quantize.detach() - input).pow(2).mean()    # get quantization loss
        quantize = input + (quantize - input).detach()      # get a new quantize instance

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        """ Returns embedding tensor for a batch of indices """
        return F.embedding(embed_id, self.embed.transpose(0, 1))    # looks up embeddings


class ResBlock(nn.Module):
    """ Residual Block """
    def __init__(self, in_channel, channel):
        """
        Initialize a Residual Block 
        
        Args:
            in_channel: Number of channels in the input data
            channel: Number of channels produced by the convolution
        """
        super().__init__()

        self.conv = nn.Sequential(      # sequantially apply a list of modules
            nn.ReLU(),                  # activation function
            nn.Conv2d(in_channel, channel, 3, padding=1),   # conv layer
            nn.ReLU(inplace=True),      # activation function
            nn.Conv2d(channel, in_channel, 1),              # conv layer
        )

    def forward(self, input):
        """
        Forward Propagation
        
        Args:
            input: Tensor of the input data
        
        Returns:
            out: Tensor of the output data
        """
        out = self.conv(input)
        out += input            # skip connection

        return out


class Encoder(nn.Module):
    """ Transform input into latent code """
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        """
        Initialize an Encoder
        
        Args:
            in_channel: Number of channels in the input
            channel: Number of channels in the convolutional layers
            n_res_block: Number of residual blocks to append
            n_res_channel: Number of channels in the residual blocks
            stride: Downsampling stride
        """
        super().__init__()      # initialize the parent nn.Module class

        if stride == 4:         # set up layers to downsample the input by a factor of 4
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:       # set up layers to downsample the input by a factor of 2
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):            # append a serials of ResBlocks
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))    # append an activation function

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        """
        Forward Propagation
        
        Args:
            input: Tensor of the input data to encode

        Returns:
            Tensor of the latent code of given input
        """
        return self.blocks(input)       # pass the input through the sequance of blocks


class Decoder(nn.Module):
    """ Take a latent code and generate an output """
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        """
        Initialize a Decoder
        
        Args:
            in_channel: Number of channels in the input
            out_channel: Number of channels in the output
            channel: Number of channels in the decoder's layers
            n_res_block: Number of residual blocks to append
            n_res_channel: Number of channels to use in the residual blocks
            stride: Upsampling stride
        """
        super().__init__()      # initialize the parent nn.Module class

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]     # start with a conv layer

        for i in range(n_res_block):                                # append a serials of ResBlocks
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))                        # append an activation function

        if stride == 4:         # appends a series of layers to upsample the tensor by a factor of 4
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:       # appends a layer to upsample the tensor by a factor of 2
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        """
        Forward Propagation
        
        Args:
            input: Tensor of the latent code to decode

        Returns:
            Tensor of the decoded data
        """
        return self.blocks(input)       # pass the input through the sequance of blocks


class VQVAE(nn.Module):
    """ Vector Quantized Variational Autoencoder """
    def __init__(
        self,
        in_channel=1,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
    ):
        """
        Initialize a VQVAE module
        
        Args:
            in_channel: Number of channels in the input tensor
            channel: Number of channels in the convolutional layers
            n_res_block: Number of residual blocks
            n_res_channel: Number of channels in the residual blocks
            embed_dim: Number of dimensions of the embedding vectors
            n_embed: Number of embedding vectors
        """
        super().__init__()      # initialize the parent nn.Module class

        # initialize two encoders with different spatial resolutions
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        # set up the quantization process for the top latent code
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = VectorQuantizer(embed_dim, n_embed)
        # initialize the decoder for the top latent code
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        # sets up the quantization process for the bottom (fine) latent code
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = VectorQuantizer(embed_dim, n_embed)
        # initialize an upsampling layer for the top latent code
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        # initialize the main decoder
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

    def forward(self, input):
        """
        Forward Propagation
        
        Args:
            input: Tensor of the input data

        Returns:
            dec: reconstructed output
            diff: quantization loss
        """
        quant_t, quant_b, diff, _, _ = self.encode(input)   # encode input into two quantized latent codes, get quantization loss
        dec = self.decode(quant_t, quant_b)                 # get reconstructed output

        return dec, diff

    def encode(self, input):
        """
        Transform input into latent code
        
        Args:
            input: Tensor of the input data

        Returns:
            quant_t: Quantized latent code for the top level
            quant_b: Quantized latent code for the bottom level
            diff_t + diff_b: Sum of the quantization losses for both levels
            id_t: Indices of the quantized vectors from the top embedding table.
            id_b: Indices of the quantized vectors from the bottom embedding table.
        """
        # encode the input using both encoders
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        # Quantization (top)
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)   # convolution and permute
        quant_t, diff_t, id_t = self.quantize_t(quant_t)            # quantize
        quant_t = quant_t.permute(0, 3, 1, 2)                       # permute back to standard format
        diff_t = diff_t.unsqueeze(0)                                # quantization loss

        # Merge
        dec_t = self.dec_t(quant_t)             # decode to the spatial resolution of bottom latent code
        enc_b = torch.cat([dec_t, enc_b], 1)    # concatenete along the channel dimension

        # Quantization (bottom)
        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)   # convolution and permute
        quant_b, diff_b, id_b = self.quantize_b(quant_b)            # quantize
        quant_b = quant_b.permute(0, 3, 1, 2)                       # permute back to standard format
        diff_b = diff_b.unsqueeze(0)                                # quantization loss

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        """
        Take a quantized latent code and generate an output
        
        Args:
            quant_t: Top level quantized latent code
            quant_b: Bottom level quantized latent code

        Returns:
            dec: Tensor of the reconstructed output
        """
        upsample_t = self.upsample_t(quant_t)       # upsample the top quantized latent code
        quant = torch.cat([upsample_t, quant_b], 1) # concatenate both latent code
        dec = self.dec(quant)                       # get the reconstructed output

        return dec

    def decode_code(self, code_t, code_b):
        """
        Take latent code indices and generate an output
        
        Args:
            code_t: top level latent code indices
            code_b: bottom level latent code indices

        Returns:
            dec: Tensor of the reconstructed output
        """
        quant_t = self.quantize_t.embed_code(code_t)    # get the top level quantized vectors
        quant_t = quant_t.permute(0, 3, 1, 2)           # permute
        quant_b = self.quantize_b.embed_code(code_b)    # get the bottom level quantized vectors
        quant_b = quant_b.permute(0, 3, 1, 2)           # permute

        dec = self.decode(quant_t, quant_b)             # get the reconstructed output

        return dec
