""" VQVAE2 Moduels """

import torch
import torch.nn as nn
import torch.nn.functional as F

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
