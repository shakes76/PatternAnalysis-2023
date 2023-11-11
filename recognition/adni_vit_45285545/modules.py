'''
Components of ViT model. Implementation adapted from PyTorch VisionTransformer.

Other references:

    [1] Y. Bazi et al., "Vision Transformers for Remote Sensing Image Classification,"
        Remote Sens., vol. 13, pp. 516-535, Feb. 2021, doi: 10.3390/rs13030516.

    [2] A. Dosovitsky et al., "An Image is Worth 16x16 Words: Transformers for
        Image Recognition at Scale", arXiv: 2010.11929 [cs.CV], 2021.

    [3] B. Pulfer. “Vision Transformers from Scratch (PyTorch): A step-by-step guide.”
        Medium.com. https://medium.com/mlearning-ai/vision-transformers-from-scratch-
        pytorch-a-step-by-step-guide-96c3313c2e0c (accessed Oct. 18, 2023).

'''
import math
from collections import OrderedDict
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
from torchvision.models.vision_transformer import MLPBlock


class EncoderBlock(nn.Module):
    '''
    Transformer encoder block. Implements Figure 2(b) from [1].

    Args:
        num_heads (int): Number of parallel attention heads per multi-head
            self attention.
        hidden_dim (int): Embedding dimension of multi-head self attention,
            also input dimension of following feedforward layer.
        mlp_dim (int): Output dimension of feedforward layer, equal to output
            dimension of whole EncoderBlock.
        dropout (float): Dropout probability, applied to feedforward.
        attention_dropout (float): Dropout probability, applied to attention.
        norm_layer (callable): Normalisation layer type, applied before both
            attention and feedforward layers to normalise layer weights.
            Defaults to LayerNorm.

    '''

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        # Number of self attention heads, for multi-head self attention (MSA)
        self.num_heads = num_heads

        # Attention block: layer norm + MSA
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block: layer norm + fully-connected layers (aka feedforward)
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f'Expected (batch_size, seq_length, hidden_dim) got {input.shape}')

        x = self.ln_1(input)
        # PyTorch MSA module returns (attention output, output weights), but we
        # do not need the output weights (have specified need_weights=False)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y) # MLP block implements its own dropout, if required
        return x + y


class Encoder(nn.Module):
    '''
    Transformer Model Encoder for sequence to sequence translation.
    Implements Figure 2(a), "Transformer Encoder", from [1].

    Args:
        seq_length (int): Number of tokens in Encoder input sequence.
        num_layers (int): Number of (attention + feedforward) layers in Encoder.
        num_heads (int): Number of parallel attention heads per multi-head self
            attention in each EncoderBlock.
        hidden_dim (int): Embedding dimension of multi-head self attention, also
            input dimension of following feedforward layer in each EncoderBlock.
        mlp_dim (int): Output dimension of feedforward layers, equal to output
            dimension of each EncoderBlock.
        dropout (float): Dropout probability, applied to feedforward layers.
        attention_dropout (float): Dropout probability, applied to attention
            layers.
        norm_layer (callable): Normalisation layer type, applied before both
            attention and feedforward layers to normalise layer weights.
            Defaults to LayerNorm.

    '''

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        # Define positional embedding for relative positions of tokens
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02)) # from BERT

        # Construct dropout layer for feedforward layers of EncoderBlocks
        self.dropout = nn.Dropout(dropout)

        # Stack encoder layers; layer names are important for loading pre-trained weights
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f'encoder_layer_{i}'] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f'Expected (batch_size, seq_length, hidden_dim) got {input.shape}')
        x = input + self.pos_embedding
        x = self.dropout(x)
        x = self.layers(x)
        x = self.ln(x)
        return x


class ViT(nn.Module):
    '''
    Vision Transformer, as presented by [2].

    Args:
        image_size (int): Side length in pixels of square input image.
        patch_size (int): Side length in pixels of square patches for converting
            image into sequence of tokens.
        num_layers (int): Number of (attention + feedforward) layers in Encoder.
        num_heads (int): Number of parallel attention ehads per multi-head self
            attention in each EncoderBlock.
        hidden_dim (int): Embedding dimension of multi-head self attention, also
            input dimension of following feedforward layer in each EncoderBlock.
        mlp_dim (int): Output dimension of feedforward layers, equal to output
            dimension of each EncoderBlock.
        dropout (float): Dropout probability, applied to feedforward layers.
            Defaults to no dropout.
        attention_dropout (float): Dropout probability, applied to attention
            layers. Defaults to no dropout.
        num_classes (int): Number of classification output classes. Defaults to
            2 classes specifically for ADNI dataset.
        norm_layer (callable): Normalisation layer type, applied before both
            attention and feedforward layers to normalise layer weights.
            Defaults to LayerNorm.

    '''

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 2,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, 'Input shape must be divisible by patch size')
        self.image_size = image_size
        self.patch_size = patch_size
        self.hiddem_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.num_classes = num_classes

        # Conv layer used to project patches before going into Transformer encoder
        self.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        # Length of transformer input sequence; i.e. number of patches in image
        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        # Transformer encoder, consisting of `num_layers` stacked
        # encoder blocks (multi-head self attention + FC)
        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        # Fully-connected classification head
        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        heads_layers['head'] = nn.Linear(hidden_dim, num_classes)
        self.heads = nn.Sequential(heads_layers)

        # Initialise patchify stem, which transforms/projects vectorised patches
        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        if self.conv_proj.bias is not None:
            nn.init.zeros_(self.conv_proj.bias)

        # Initialise FC classification head
        nn.init.zeros_(self.heads.head.weight)
        nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        '''Patchify input images into vectorised patches.'''
        n, _, h, w = x.shape # channels get incorporated into patches
        p = self.patch_size
        torch._assert(h == self.image_size, f'Wrong image height! Expected {self.image_size} but got {h}')
        torch._assert(w == self.image_size, f'Wrong image width! Expected {self.image_size} but got {w}')
        n_h = h // p # image height divides into n_h patches
        n_w = w // p # image width divides into n_w patches

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w); i.e. patchify
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hiddem_dim, (n_h * n_w)); i.e. vectorise
        x = x.reshape(n, self.hiddem_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # From PyTorch: "The self attention layer expects inputs in the format
        # (N, S, E) where S is the source sequence length, N is the batch size,
        # E is the embedding dimension"
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        # by feeding trained "class" embedding into MLP classification head
        x = x[:, 0]
        x = self.heads(x)

        return x
