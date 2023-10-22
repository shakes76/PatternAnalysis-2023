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


class CrossAttention(nn.Module):

    def __init__(self, dim_latent, heads):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(dim_latent, heads)
        self.linear = nn.Linear(dim_latent, dim_latent)

    def forward(self, kv, q):
        out, _ = self.attention(q, kv, kv)
        out = self.linear(out)
        return out
    

class PerceiverBlock(nn.Module):

    def __init__(self, dim_latent, heads, transformer_depth):
        self.cross_attention = CrossAttention(dim_latent, heads)
        self.transformer = nn.Transformer(
            d_model = dim_latent, 
            nhead = heads,
            num_encoder_layers = transformer_depth,
            num_decoder_layers = transformer_depth
        )

    def forward(self, x, y):
        out = self.cross_attention(x, y)
        out = self.transformer(out)
        return out
    

class Classifier(nn.Module):

    def __init__(self, dim_latent, n_classes):
        super(Classifier, self).__init__()
        self.output_layer = nn.Linear(dim_latent, n_classes)

    def forward(self, x):
        averaged = torch.mean(x, dim = 2)
        out = self.output_layer(averaged)
        return out
    

class Perceiver(nn.Module):
    """
    Architecture and layout of Perceiver adapted from source:
    https://medium.com/@curttigges/the-annotated-perceiver-74752113eefb

    The perceiver model constructed here has a few modifications. Additionally,
    we use the pytorch transformer model as it 
    """

    def __init__(self, dim_latent, heads, transformer_depth, num_blocks, n_classes):
        super(Perceiver, self).__init__()

        self.pe = PositionalEncoding(240 * 240)

        latent = torch.zeros((dim_latent, 1, 1))
        torch.nn.init.trunc_normal(latent, mean = 0, std = 0.02, a = -2, b = 2)
        self.latent = nn.Parameter(latent)

        self.blocks = nn.ModuleList(
            [PerceiverBlock(dim_latent, heads, transformer_depth) for _ in range(num_blocks)]
        )
        self.classifier = Classifier(dim_latent, n_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        
        out = self.latent.reshape((-1, batch_size, -1))
        embedded = self.pe(x)

        for block in self.blocks:
            out = block(embedded, out)

        



