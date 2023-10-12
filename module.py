import torch
import torch.nn as nn

class PositionalImageEmbedding(nn.Module):
    """
        Initialise the PositionalImageEmbedding module.

        Args:
            input_channels (int): Number of input channels in the image.
            embed_dim (int): Dimension of the embedded image representation.
            bands (int): Number of Fourier feature bands for positional encoding.
        
        
        Returns:
            (Tensor): Embedded image representation of shape (HEIGHT*WIDTH x BATCH_SIZE x EMBED_DIM).
        """
    def __init__(self, input_channels, embed_dim, bands=4):
        super().__init__()

        # Initialise the Fourier features for positional encoding
        self.ff = self.fourier_features(bands)

        # Initialise a 1D convolution layer to process the concatenated input
        self.conv = nn.Conv1d(input_channels + self.ff.shape[0], embed_dim, 1)

    def forward(self, x):
        """
        Forward pass of the PositionalImageEmbedding module.

        Args:
            x (Tensor): Input image tensor of shape (BATCH_SIZE x CHANNELS x HEIGHT x WIDTH).

        Returns:
            (Tensor): Embedded image representation of shape (HEIGHT*WIDTH x BATCH_SIZE x EMBED_DIM).
        """
        # Create position encoding with the same shape as the input
        enc = self.ff.unsqueeze(0).expand(x.shape[0], -1, -1).type_as(x)

        # Concatenate the position encoding along the channel dimension
        x = torch.cat([x, enc], dim=1)

        # Flatten the input
        x = x.flatten(2)

        # Apply 1D convolution
        x = self.conv(x)

        # Permute the dimensions for the final output
        x = x.permute(2, 0, 1)

        return x

class PerceiverAttentionBlock(nn.Module):
    """
    Perceiver Attention Block

    This module is used for both cross-attention and the latent transformer.

    Args:
        embed_dim (int): Dimension of the embedded representations.
        mlp_dim (int): Dimension of the feedforward network hidden layer.
        n_heads (int): Number of attention heads.
        dropout (float, optional): Dropout probability. Default is 0.0.

    Inputs:
        latent (Tensor): The query tensor of shape [LATENT_DIM x BATCH_SIZE x EMBED_DIM].
        image (Tensor): The key and value tensor of shape [PIXELS x BATCH_SIZE x EMBED_DIM].

    Outputs:
        Tensor: The output tensor of shape [LATENT_DIM x BATCH_SIZE x EMBED_DIM].

    """

    def __init__(self, embed_dim, mlp_dim, n_heads, dropout=0.0):
        super().__init__()

        # Layer Normalization for the image
        self.lnorm1 = nn.LayerNorm(embed_dim)

        # Multi-Head Self-Attention
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads)

        # Layer Normalization for the output of the attention
        self.lnorm2 = nn.LayerNorm(embed_dim)

        # First linear layer
        self.linear1 = nn.Linear(embed_dim, mlp_dim)

        # GELU activation function
        self.act = nn.GELU()

        # Second linear layer
        self.linear2 = nn.Linear(mlp_dim, embed_dim)

        # Dropout layer
        self.drop = nn.Dropout(dropout)

    def forward(self, latent, image):
        """
        Forward pass of the Perceiver Attention Block.

        Args:
            latent (Tensor): The query tensor of shape [LATENT_DIM x BATCH_SIZE x EMBED_DIM].
            image (Tensor): The key and value tensor of shape [PIXELS x BATCH_SIZE x EMBED_DIM].

        Returns:
            Tensor: The output tensor of shape [LATENT_DIM x BATCH_SIZE x EMBED_DIM].

        """

        # Layer normalization and self-attention
        out = self.lnorm1(image)
        out, _ = self.attn(query=latent, key=image, value=image)

        # Compute the first residual connection
        resid = out + latent

        # Layer normalization and feedforward network
        out = self.lnorm2(resid)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        out = self.drop(out)

        # Compute the second residual connection
        out = out + resid

        return out
    
class LatentTransformer(nn.Module):
    """
    Latent Transformer module with multiple decoder layers.

    Args:
        embed_dim (int): Dimension of the embedded representations.
        mlp_dim (int): Dimension of the feedforward network hidden layer.
        n_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        n_layers (int): Number of decoder layers.
        
    Returns:
            Tensor: Transformed latent tensor of shape [LATENT_DIM x BATCH_SIZE x EMBED_DIM].


    """

    def __init__(self, embed_dim, mlp_dim, n_heads, dropout, n_layers):
        super().__init__()

        # Create a list of decoder layers (PerceiverAttention blocks)
        self.transformer = nn.ModuleList([
            PerceiverAttentionBlock(
                embed_dim=embed_dim, 
                mlp_dim=mlp_dim, 
                n_heads=n_heads, 
                dropout=dropout) 
            for _ in range(n_layers)
        ])

    def forward(self, l):
        """
        Forward pass of the LatentTransformer module.

        Args:
            l (Tensor): Latent tensor of shape [LATENT_DIM x BATCH_SIZE x EMBED_DIM].

        Returns:
            Tensor: Transformed latent tensor of shape [LATENT_DIM x BATCH_SIZE x EMBED_DIM].

        """
        for transform in self.transformer:
            l = transform(l, l)
        return l

class PerceiverBlock(nn.Module):
    """
    Block consisting of one cross-attention layer and one latent transformer.

    Args:
        embed_dim (int): Dimension of the embedded representations.
        attn_mlp_dim (int): Dimension of the cross-attention's feedforward network hidden layer.
        trnfr_mlp_dim (int): Dimension of the latent transformer's feedforward network hidden layer.
        trnfr_heads (int): Number of attention heads for the latent transformer.
        dropout (float): Dropout probability.
        trnfr_layers (int): Number of layers in the latent transformer.

    """

    def __init__(self, embed_dim, attn_mlp_dim, trnfr_mlp_dim, trnfr_heads, dropout, trnfr_layers):
        super().__init()
        
        # Cross-Attention layer
        self.cross_attn = PerceiverAttention(
            embed_dim, attn_mlp_dim, n_heads=1, dropout=dropout)

        # Latent Transformer module
        self.latent_transformer = LatentTransformer(
            embed_dim, trnfr_mlp_dim, trnfr_heads, dropout, trnfr_layers)

    def forward(self, x, l):
        """
        Forward pass of the PerceiverBlock module.

        Args:
            x (Tensor): Input tensor of shape [PIXELS x BATCH_SIZE x EMBED_DIM].
            l (Tensor): Latent tensor of shape [LATENT_DIM x BATCH_SIZE x EMBED_DIM].

        Returns:
            Tensor: Transformed latent tensor of shape [LATENT_DIM x BATCH_SIZE x EMBED_DIM].

        """
        # Apply cross-attention on the input and latent tensor
        l = self.cross_attn(x, l)

        # Apply the latent transformer
        l = self.latent_transformer(l)

        return l



