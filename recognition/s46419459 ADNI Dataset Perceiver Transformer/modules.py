import torch
import torch.nn as nn

import math


class PositionalEncoding(nn.Module):
    """
    PositionalEncoding layer sourced from the following source:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class CrossAtttention(nn.Module):

    def __init__(self, dim_latent, heads):
        super(CrossAtttention, self).__init__()
        self.attention = nn.MultiheadAttention(dim_latent, heads)
        self.linear = nn.Linear(dim_latent, dim_latent)

    def forward(self, kv, q):
        out, _ = self.attention(q, kv, kv)
        out = self.linear(out)
        return out
    

class PerceiverBlock(nn.Module):

    def __init__(self, dim_latent, heads, transformer_depth):
        self.cross_attention = CrossAtttention(dim_latent, heads)
        self.transformer = nn.Transformer(
            d_model = dim_latent, 
            nhead = heads,
            num_encoder_layers = transformer_depth,
            num_decoder_layers = transformer_depth
        )

    def forward(self, x):
        out = self.cross_attention(x)
        out = self.transformer(out)
        return out
    

class Classifer(nn.Module):

    def __init__(self, dim_latent, n_classes):
        self.output_layer = nn.Linear(dim_latent, n_classes)

    def forward(self, x):
        averaged = torch.mean(x, dim = 2)
        out = self.output_layer(averaged)
        return out
