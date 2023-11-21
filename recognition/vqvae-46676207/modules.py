""" VQVAE2 Moduels """

import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from functools import partial, lru_cache

import numpy as np


class WNConv2d(nn.Module):
    """
    A 2D convolutional layer with weight normalization.
    """
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
        """
        Initialize an instance of the WNConv2d class.

        Args:
            in_channel: Number of input channels
            out_channel: Number of output channels
            kernel_size: Size of the kernel
            stride: Stride of the convolution
            padding: Zero padding for the input
            bias: Whether to add a bias term to output
            activation: Activation function
        """
        super().__init__()

        self.conv = nn.utils.weight_norm(   # initialize a 2D convolutional layer
            nn.Conv2d(                      # with weight normalization
                in_channel,
                out_channel,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        )

        self.out_channel = out_channel

        if isinstance(kernel_size, int):                # check if kernel_size is an integer
            kernel_size = [kernel_size, kernel_size]    # and then convert to a list

        self.kernel_size = kernel_size

        self.activation = activation

    def forward(self, input):
        """
        Forward pass

        Args:
            input: Input tensor to pass through the layer

        Return:
            out: The tensor processed by the convolution and activation function
        """
        out = self.conv(input)              # pass through the conv layer

        if self.activation is not None:
            out = self.activation(out)      # apply the activation function

        return out                          # return the result


class CausalConv2d(nn.Module):
    """
    A causal convolutional layer
    """
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding='downright',
        activation=None,
    ):
        """
        Initialize an instance of the CausalConv2d class

        Args:
            in_channel: Number of input channels
            out_channel: Number of output channels
            kernel_size: Size of the convolutional kernel
            stride: Stride of the convolution
            padding: Type of padding to be used (downright/down/causal)
            activation: Optional activation function
        """
        super().__init__()

        if isinstance(kernel_size, int):        # convert kernel_size to list if is integer
            kernel_size = [kernel_size] * 2

        self.kernel_size = kernel_size

        if padding == 'downright':              # pad at top & left
            pad = [kernel_size[1] - 1, 0, kernel_size[0] - 1, 0]

        elif padding == 'down' or padding == 'causal':  # pad top & left & right
            pad = kernel_size[1] // 2

            pad = [pad, pad, kernel_size[0] - 1, 0]

        self.causal = 0
        if padding == 'causal':                 # set causal to half the kernel width
            self.causal = kernel_size[1] // 2

        self.pad = nn.ZeroPad2d(pad)            # zero-padding layer

        self.conv = WNConv2d(                   # initialize the conv layer
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
            padding=0,
            activation=activation,
        )

    def forward(self, input):
        """
        Forward pass

        Args:
            input: The input tensor

        Returns:
            out: The output tensor after applying the causal convolution
        """
        out = self.pad(input)   # pad the input tensor

        if self.causal > 0:     # zero out specific weights
            self.conv.conv.weight_v.data[:, :, -1, self.causal :].zero_()

        out = self.conv(out)    # pass through the conv layer

        return out              # return the result tensor


class GatedResBlock(nn.Module):
    """
    Gated Residual Block
    """
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
        """
        Initialize an instance of GatedResBlock

        Args:
            in_channel: Number of input channels
            channel: Number of channels in the intermediate layers
            kernel_size: Size of the convolutional kernel
            conv: Type of convolution (wnconv2d/causal_downright/causal)
            activation: Activation function
            dropout: Dropout rate
            auxiliary_channel: Number of channels in the auxiliary input
            condition_dim: Dimension of the conditional input
        """
        super().__init__()

        match conv:                                                     # choose conv layer
            case 'wnconv2d':
                conv_module = partial(WNConv2d, padding=kernel_size // 2)

            case 'causal_downright':
                conv_module = partial(CausalConv2d, padding='downright')

            case 'causal':
                conv_module = partial(CausalConv2d, padding='causal')

        self.activation = activation()                                  # activation func
        self.conv1 = conv_module(in_channel, channel, kernel_size)      # 1st conv layer

        if auxiliary_channel > 0:
            self.aux_conv = WNConv2d(auxiliary_channel, channel, 1)     # auxiliary conv

        self.dropout = nn.Dropout(dropout)                              # dropout layer

        self.conv2 = conv_module(channel, in_channel * 2, kernel_size)  # 2nd conv layer

        if condition_dim > 0:
            self.condition = WNConv2d(condition_dim, in_channel * 2, 1, bias=False) # conditional conv

        self.gate = nn.GLU(1)                                           # gated linuear unit

    def forward(self, input, aux_input=None, condition=None):
        """
        Forward pass

        Args:
            input: Input tensor
            aux_input: Auxiliary input tensor
            condition: Conditional tensor

        Returns:
            out: Output tensor processed through the gated residual block
        """
        out = self.conv1(self.activation(input))                    # apply conv layer

        if aux_input is not None:
            out = out + self.aux_conv(self.activation(aux_input))   # apply activation & auxiliary conv

        out = self.activation(out)                                  # apply activation
        out = self.dropout(out)                                     # apply dropout
        out = self.conv2(out)                                       # apply 2nd conv layer

        if condition is not None:
            condition = self.condition(condition)                   # apply the conditional conv
            out += condition

        out = self.gate(out)                                        # apply gating
        out += input

        return out                                                  # return result


@lru_cache(maxsize=64)
def causal_mask(size):
    """
    Create and return causal mask

    Args:
        size: Size of the mask

    Returns:
        A tuple contains a causal mask and a start mask
    """
    shape = [size, size]                                        # Def mask shape
    mask = np.triu(np.ones(shape), k=1).astype(np.uint8).T      # Upper triangular matrix
    start_mask = np.ones(size).astype(np.float32)               # Mask of ones
    start_mask[0] = 0                                           # Set first position to zero

    return (
        torch.from_numpy(mask).unsqueeze(0),                    # return mask
        torch.from_numpy(start_mask).unsqueeze(1),              # return start
    )


def wn_linear(in_dim, out_dim):
    """
    Create and return a weight-normalized linear layer

    Args:
        in_dim: Input dimension for the linear layer
        out_dim: Output dimension for the linear layer
    
    Returns:
        A weight-normalized linear layer
    """
    return nn.utils.weight_norm(nn.Linear(in_dim, out_dim))     # return a weight-normalized linear layer


class CausalAttention(nn.Module):
    """
    Causal multi-head attention mechanism
    """
    def __init__(self, query_channel, key_channel, channel, n_head=8, dropout=0.1):
        """
        Initialize an instance of CausalAttention class

        Args:
            query_channel: Number of channels in the query
            key_channel: Number of channels in the key
            channel: Total number of channels for the query, key, and value after transformations
            n_head: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
                                                        # Initializes a weight-normalized linear layer
        self.query = wn_linear(query_channel, channel)  # for query transformation
        self.key = wn_linear(key_channel, channel)      # for key transformation
        self.value = wn_linear(key_channel, channel)    # for value transformation

        self.dim_head = channel // n_head               # get number of channels per attention head
        self.n_head = n_head

        self.dropout = nn.Dropout(dropout)              # dropout layer

    def forward(self, query, key):
        """
        Forward pass

        Args:
            query: Query tensor
            key: Key tensor
        Returns:
            out: The tensor after applying the causal attention mechanism
        """
        batch, _, height, width = key.shape                                             # get batch size, height, and width

        def reshape(input):
            return input.view(batch, -1, self.n_head, self.dim_head).transpose(1, 2)    # reshape tensor for multi head attention processing

        query_flat = query.view(batch, query.shape[1], -1).transpose(1, 2)              # flatten & transpose query tensor
        key_flat = key.view(batch, key.shape[1], -1).transpose(1, 2)                    # flatten & transpose key tensor
        query = reshape(self.query(query_flat))                                         # transform & reshape query tensor
        key = reshape(self.key(key_flat)).transpose(2, 3)                               # transform & reshape key tensor
        value = reshape(self.value(key_flat))                                           # transform & reshape value tensor

        attn = torch.matmul(query, key) / sqrt(self.dim_head)                           # get normalized raw attention scores
        mask, start_mask = causal_mask(height * width)                                  # get causal mask and start mask
        mask = mask.type_as(query)
        start_mask = start_mask.type_as(query)
        attn = attn.masked_fill(mask == 0, -1e4)                                        # apply causal mask
        attn = torch.softmax(attn, 3) * start_mask                                      # apply softmax & times start mask
        attn = self.dropout(attn)                                                       # apply dropout to attention score

        out = attn @ value                                                              # get output tensor
        out = out.transpose(1, 2).reshape(                                              # reshape & permute output tensor
            batch, height, width, self.dim_head * self.n_head
        )
        out = out.permute(0, 3, 1, 2)

        return out                                                                      # return output tensor


class PixelBlock(nn.Module):
    """
    A block of PixelSNAIL architecture
    """
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
        """
        Initialize an instance of PixelBlock class

        Args:
            in_channel: Number of input channels
            channel: Number of internal channels for the GatedResBlocks
            kernel_size: Size of the kernel
            n_res_block: Number of GatedResBlocks
            attention: Whether to use causal attention
            dropout: Dropout rate
            condition_dim: Dimension of the conditioning variable
        """
        super().__init__()

        resblocks = []                              # initialize a list of GatedResBlocks
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

        self.resblocks = nn.ModuleList(resblocks)   # convert to a ModuleList

        self.attention = attention                  # Attention flag

        if attention:                                                                   # Create additional layers if attention is enabled
            self.key_resblock = GatedResBlock(                                          # Gated residual block for key trans
                in_channel * 2 + 2, in_channel, 1, dropout=dropout
            )
            self.query_resblock = GatedResBlock(                                        # Gated residual block for query trans
                in_channel + 2, in_channel, 1, dropout=dropout
            )

            self.causal_attention = CausalAttention(                                    # Causal attention module
                in_channel + 2, in_channel * 2 + 2, in_channel // 2, dropout=dropout
            )

            self.out_resblock = GatedResBlock(                                          # Gated residual block for output
                in_channel,
                in_channel,
                1,
                auxiliary_channel=in_channel // 2,
                dropout=dropout,
            )

        else:                                                                           # if attention isn't enabled
            self.out = WNConv2d(in_channel + 2, in_channel, 1)                          # Create a conv layer

    def forward(self, input, background, condition=None):
        """
        Forward pass

        Args:
            input: Input tensor
            background: Background tensor
            condition: Conditioning tensor

        Returns:
            out: Tensor after being processed by the PixelBlock
        """
        out = input                                             # start with the input

        for resblock in self.resblocks:                         # apply GatedResBlocks
            out = resblock(out, condition=condition)

        if self.attention:                                      # if attention is enabled
            key_cat = torch.cat([input, out, background], 1)    # get key tensor
            key = self.key_resblock(key_cat)
            query_cat = torch.cat([out, background], 1)         # get query tensor
            query = self.query_resblock(query_cat)
            attn_out = self.causal_attention(query, key)        # apply causal attention
            out = self.out_resblock(out, attn_out)              # Combine out with attn_out

        else:                                                   # if attention isn't enabled
            bg_cat = torch.cat([out, background], 1)            # Concatenate out with background
            out = self.out(bg_cat)                              # apply conv layer

        return out                                              # return result (out tensor)


class CondResNet(nn.Module):
    """
    Conditional Residual Network
    """
    def __init__(self, in_channel, channel, kernel_size, n_res_block):
        """
        Initialize an instance of CondResNet class

        Args:
            in_channel: Number of input channels
            channel: Number of internal channels for convolution and GatedResBlocks
            kernel_size: Size of the kernel
            n_res_block: Number of GatedResBlocks
        """
        super().__init__()

        blocks = [WNConv2d(in_channel, channel, kernel_size, padding=kernel_size // 2)] # weighted-normalized conv layer

        for i in range(n_res_block):
            blocks.append(GatedResBlock(channel, channel, kernel_size))                 # append GatedResBlocks to blocks list

        self.blocks = nn.Sequential(*blocks)                                            # convert to a sequential model

    def forward(self, input):
        """
        Forward pass

        Args:
            input: Input tensor
        Returns:
            out: Tensor after being processed by the CondResNet
        """
        return self.blocks(input)


def shift_down(input, size=1):
    """
    Shift tensors downward

    Args:
        input: Input tensor
        size: Number of positions to shift
    Returns:
        Tensor after being shifted
    """
    return F.pad(input, [0, 0, size, 0])[:, :, : input.shape[2], :] # pad at the top, shift downward, slice to the original height


def shift_right(input, size=1):
    """
    Shift tensors to the right

    Args:
        input: Input tensor
        size: Number of positions to shift
    Returns:
        Tensor after being shifted
    """
    return F.pad(input, [size, 0, 0, 0])[:, :, :, : input.shape[3]] # pad at the left, shift to the right, slice to the original width


class PixelSNAIL(nn.Module):
    """
    Defines the PixelSNAIL architecture
    """
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
        """
        Initialize an instance of the PixelSNAIL class

        Args:
            shape: Shape of input
            n_class: Number of classes
            channel: Number of channels in conv layer
            kernel_size: Size of the kernel
            n_block: Number of PixelBlocks
            n_res_block: Number of GatedResBlocks in each PixelBlock
            res_channel: Number of channels in the conv layers in residual blocks
            attention: Whether to use the attention mechanism
            dropout: Dropout rate
            n_cond_res_block: Number of conditional residual blocks
            cond_res_channel: Number of channels in the conv layers in conditional residual blocks
            cond_res_kernel: Size of the kernel in conditional residual blocks
            n_out_res_block: Number of residual blocks in the output sequnce
        """
        super().__init__()

        height, width = shape               # Extract height & weight

        self.n_class = n_class

        if kernel_size % 2 == 0:
            kernel = kernel_size + 1        # adjust kernel size if even

        else:
            kernel = kernel_size

        self.horizontal = CausalConv2d(
            n_class, channel, [kernel // 2, kernel], padding='down'                 # horizontal causal conv layers
        )
        self.vertical = CausalConv2d(
            n_class, channel, [(kernel + 1) // 2, kernel // 2], padding='downright' # vertical causal conv layers
        )

        # Positional Coordinates
        coord_x = (torch.arange(height).float() - height / 2) / height              # tensor of relative x-coord of each pixel
        coord_x = coord_x.view(1, 1, height, 1).expand(1, 1, height, width)
        coord_y = (torch.arange(width).float() - width / 2) / width                 # tensor of relative y-coord of each pixel
        coord_y = coord_y.view(1, 1, 1, width).expand(1, 1, height, width)
        self.register_buffer('background', torch.cat([coord_x, coord_y], 1))        # register a buffer for coordinates

        self.blocks = nn.ModuleList()                                               # list of PixelBlocks

        for i in range(n_block):                                                    # append PixelBlocks to the list
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

        if n_cond_res_block > 0:                                                    # initialize conditional residual network if required
            self.cond_resnet = CondResNet(
                n_class, cond_res_channel, cond_res_kernel, n_cond_res_block
            )

        out = []                                                                    # list of final output layers

        for i in range(n_out_res_block):
            out.append(GatedResBlock(channel, res_channel, 1))                      # append GatedResBlocks to the list

        out.extend([nn.ELU(inplace=True), WNConv2d(channel, n_class, 1)])           # add activation & weight-normalized conv layer

        self.out = nn.Sequential(*out)                                              # convert to sequential module

    def forward(self, input, condition=None, cache=None):
        """
        Forward pass

        Args:
            input: Input tensor
            condition: Conditioning tensor
            cache: Intermediate tensors

        Returns:
            out: Output tensor (generated output)
            cache: Cached tensors
        """
        if cache is None:
            cache = {}                                                                  # initialize empty cache if cache isn't provided
        batch, height, width = input.shape                                              # extract batch size, height, width
        input = (
            F.one_hot(input, self.n_class).permute(0, 3, 1, 2).type_as(self.background) # encode input to one-hot format
        )
        horizontal = shift_down(self.horizontal(input))                                 # apply horizontal causal conv & shift down
        vertical = shift_right(self.vertical(input))                                    # apply vertical causal conv & shift down
        out = horizontal + vertical                                                     # combine the outputs

        background = self.background[:, :, :height, :].expand(batch, 2, height, width)  # a slice of background tensor (for positional info)

        if condition is not None:
            if 'condition' in cache:                                            # if conditioned tensor is cached
                condition = cache['condition']                                  # use it
                condition = condition[:, :, :height, :]

            else:                                                               # if confitioned tensor isn't cached
                condition = (                                                   # process one for future use
                    F.one_hot(condition, self.n_class)
                    .permute(0, 3, 1, 2)
                    .type_as(self.background)
                )
                condition = self.cond_resnet(condition)
                condition = F.interpolate(condition, scale_factor=2)
                cache['condition'] = condition.detach().clone()
                condition = condition[:, :, :height, :]

        for block in self.blocks:
            out = block(out, background, condition=condition)                   # process the output tensor through each PixelBlock

        out = self.out(out)                                                     # process the output tensor through output layers

        return out, cache                                                       # return the generated output & cache


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
