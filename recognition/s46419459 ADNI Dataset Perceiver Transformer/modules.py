import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, d_latent):
        super(MLP, self).__init__()
        self.layernorm = nn.LayerNorm(d_latent)
        self.linear1 = nn.Linear(d_latent, d_latent)
        self.linear2 = nn.Linear(d_latent, d_latent)
        self.activation = nn.GELU()

    def forward(self, x):
        out = self.layernorm(x)
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.activation(out)
        return out

class CrossAttention(nn.Module):

    def __init__(self, d_latent, n_latent, heads):
        super(CrossAttention, self).__init__()
        self.normalize = nn.LayerNorm(n_latent)
        self.attention = nn.MultiheadAttention(n_latent, heads)

    def forward(self, kv, q):
        out, _ = self.attention(q, kv, kv)
        out = self.linear(out)
        return out
    
    
class PerceiverBlock(nn.Module):

    def __init__(self, dim_latent, heads, transformer_depth):
        super(PerceiverBlock, self).__init__()
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

    def __init__(self, dim_latent, batch_size, heads, transformer_depth, num_blocks, n_classes):
        super(Perceiver, self).__init__()

        latent = torch.zeros((1, 240*240, dim_latent))
        torch.nn.init.trunc_normal_(latent, mean = 0, std = 0.02, a = -2, b = 2)
        self.latent = nn.Parameter(latent)

        self.blocks = nn.ModuleList(
            [PerceiverBlock(dim_latent, heads, transformer_depth) for _ in range(num_blocks)]
        )
        self.classifier = Classifier(dim_latent, n_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        
        out = self.latent.repeat(batch_size, 1, 1)
        embedded = x

        for block in self.blocks:
            out = block(embedded, out)

        out = self.classifier(out)
        return out



