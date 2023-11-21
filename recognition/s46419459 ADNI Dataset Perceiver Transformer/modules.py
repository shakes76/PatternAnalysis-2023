import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, embed_dim):
        super(MLP, self).__init__()
        # MLP structure inferred from paper
        self.layernorm = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.GELU()
        # Dropout was used in paper, but found to degrade results
        # so it was excluded

    def forward(self, x):
        # Attention output passed straight through MLP
        out = self.layernorm(x)
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.activation(out)
        return out


class CrossAttention(nn.Module):

    def __init__(self, embed_dim, heads):
        super(CrossAttention, self).__init__()
        # Normalization -> Attention -> MLP as suggested in paper
        self.normalize = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, heads)
        self.mlp = MLP(embed_dim)

    def forward(self, kv, q):
        # here, kv is the embedded image with size:
        # [pixels x batch_size x embed_dim]
        normalQ = self.normalize(q)
        out, _ = self.attention(normalQ, kv, kv)
        out = self.mlp(out)
        return out
    
    
class PerceiverBlock(nn.Module):

    def __init__(self, embed_dim, heads, transformer_depth):
        super(PerceiverBlock, self).__init__()
        # Each block in the perceiver is made up of 1 cross attention
        # and one transformer. The transformer used was the pytorch
        # implementation.
        self.cross_attention = CrossAttention(embed_dim, heads)
        self.transformer = nn.Transformer(
            d_model = embed_dim, 
            nhead = heads,
            num_encoder_layers = transformer_depth,
            num_decoder_layers = transformer_depth
        )

    def forward(self, x, y):
        out = self.cross_attention(x, y)
        out = self.transformer(out, out)
        return out
    

class Classifier(nn.Module):

    def __init__(self, embed_dim, n_classes):
        super(Classifier, self).__init__()
        self.linear_pass = nn.Linear(embed_dim, embed_dim)
        self.output_layer = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        # result of batch is averaged before being passed through final layer
        out = self.linear_pass(x)
        out = torch.mean(out, dim = 0)
        out = self.output_layer(out)
        return out
    

class Perceiver(nn.Module):
    """
    Architecture and layout of Perceiver adapted from source:
    https://medium.com/@curttigges/the-annotated-perceiver-74752113eefb

    The perceiver model constructed here has a few modifications.
    """

    def __init__(self, d_latent, embed_dim, heads, transformer_depth, n_perceiver_blocks, n_classes, batch_size):
        super(Perceiver, self).__init__()

        # Initialize the latent array to be a normally distributed 
        # array of parameters of size [batch_size x d_latent x embed_dim]
        # with values bound between [-2, 2]
        latent = torch.zeros((batch_size, d_latent, embed_dim))
        torch.nn.init.trunc_normal_(latent, mean = 0, std = 0.02, a = -2, b = 2)
        self.latent = nn.Parameter(latent)

        # Input of height*width size passes through the embed_dim bottleneck
        self.embed_layer = nn.Linear(240 * 240, embed_dim)

        # Stack of perceiver blocks
        self.blocks = nn.ModuleList(
            [PerceiverBlock(embed_dim, heads, transformer_depth) for _ in range(n_perceiver_blocks)]
        )

        # Classification layer
        self.classifier = Classifier(embed_dim, n_classes)

    def forward(self, x):
        
        # Rearrange tensor for attention / transformer modules
        # that expect shapes of: 
        # latent array -> [latent_dim x batch_size x embed_dim] 
        # key and val -> [pixels x batch_size x embed_dim]

        embedded = self.embed_layer(x).permute(1, 0, 2)
        out = self.latent.permute(1, 0, 2)

        for block in self.blocks:
            out = block(embedded, out)

        out = self.classifier(out)
        return out



