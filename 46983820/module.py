import math
import torch
import torch.nn as nn
import numpy as np

"""
Perceiver Transformer Model for Alzheimer's Disease Classification

Architecture and Modules adapted from:
    - https://arxiv.org/abs/2103.03206
    - https://medium.com/@curttigges/the-annotated-perceiver-74752113eefb

"""

class PositionalImageEmbedding(nn.Module):
    """
        Initialise the PositionalImageEmbedding module.

        Args:
            input_shape (tuple): Shape of the input images (e.g., (256, 256)).
            input_channels (int): Number of input channels in the image.
            embed_dim (int): Dimension of the embedded image representation.
            bands (int): Number of Fourier feature bands for positional encoding.
        
        
        Returns:
            (Tensor): Embedded image representation of shape (HEIGHT*WIDTH x BATCH_SIZE x EMBED_DIM).
        """
    def __init__(self, input_shape, input_channels, embed_dim, bands=4):
        super().__init__()

        # Initialise the Fourier features for positional encoding
        self.ff = self.fourier_features(input_shape, bands)
        # Initialise a 1D convolution layer to process the concatenated input
        self.conv = nn.Conv1d(input_channels + self.ff.shape[0], embed_dim, 1)
    
    def fourier_features(self, shape, bands):
        """
        Compute Fourier features for positional encoding.
        
        Args:
            shape (tuple): Shape of the input images (e.g., (256, 256)).
            bands (int): Number of Fourier feature bands for positional encoding.
            
        Returns:
            (Tensor): Fourier features of shape (BANDS*2 x HEIGHT x WIDTH).
            
        Source:
            Fourier features encoding adapted from:
            https://github.com/louislva/deepmind-perceiver/blob/master/perceiver/positional_encoding.py
        """
        # Create a meshgrid of coordinates
        pos = torch.stack(list(torch.meshgrid(*(torch.linspace(-1.0, 1.0, steps=n) for n in list(shape)))))
        pos = pos.unsqueeze(0).expand((bands,) + pos.shape)

        # Compute the frequency of the Fourier features
        log_start = math.log(1.0)
        log_end = math.log(shape[0] / 2)
        band_frequencies = torch.logspace(log_start, log_end, steps=bands, base=math.e)
        band_frequencies = band_frequencies.view((bands,) + tuple(1 for _ in pos.shape[1:]))
        band_frequencies = band_frequencies.expand(pos.shape)
        
        # Compute the Fourier features
        result = (band_frequencies * math.pi * pos).view((len(shape) * bands,) + shape)

        # Use both Sin and Cos for each band and add raw position
        result = torch.cat([torch.sin(result),torch.cos(result),], dim=0,)

        return result

    def forward(self, x):
        """
        Forward pass of the PositionalImageEmbedding module.

        Args:
            x (Tensor): Input image tensor of shape (BATCH_SIZE x CHANNELS x HEIGHT x WIDTH).

        Returns:
            (Tensor): Embedded image representation of shape (HEIGHT*WIDTH x BATCH_SIZE x EMBED_DIM).
        """
        # Concatenate the input image with the Fourier features
        enc = self.ff.unsqueeze(0).expand((x.shape[0],) + self.ff.shape)
        enc = enc.type_as(x)
        x = torch.cat([x, enc], dim=1)

        # (BATCH_SIZE x CHANNELS x HEIGHT*WIDTH)
        x = x.flatten(2)
        
        # (BATCH_SIZE x EMBED_DIM x HEIGHT*WIDTH)
        x = self.conv(x)

        # (HEIGHT*WIDTH x BATCH_SIZE x EMBED_DIM)
        return x.permute(2, 0, 1)

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
        (Tensor): The output tensor of shape [LATENT_DIM x BATCH_SIZE x EMBED_DIM].

    """

    def __init__(self, embed_dim, mlp_dim, n_heads, dropout=0.0):
        super().__init__()

        # Layer Normalization for the image and for the output of the attention
        self.layer_normalisation = nn.LayerNorm(embed_dim)

        # Multi-Head Self-Attention
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads)

        # First linear layer
        self.linear1 = nn.Linear(embed_dim, mlp_dim)

        # GELU activation function
        self.activation = nn.GELU()

        # Second linear layer
        self.linear2 = nn.Linear(mlp_dim, embed_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, in1, in2):
        """
        Forward pass of the Perceiver Attention Block.

        Args:
            in1 (Tensor): The query tensor of shape [LATENT_DIM x BATCH_SIZE x EMBED_DIM].
            in2 (Tensor): The key and value tensor of shape [PIXELS x BATCH_SIZE x EMBED_DIM].

        Returns:
            (Tensor): The output tensor of shape [LATENT_DIM x BATCH_SIZE x EMBED_DIM].

        """

        # Layer normalization and self-attention
        latent = self.layer_normalisation(in2)
        latent, _ = self.attention(query=in2, key=in1, value=in1)

        # Compute the first residual connection
        resid = latent + in2

        # Layer normalization and feedforward network
        latent = self.layer_normalisation(resid)
        latent = self.linear1(latent)
        latent = self.activation(latent)
        latent = self.linear2(latent)
        latent = self.dropout(latent)

        # Compute the second residual connection
        latent += resid

        return latent
    
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
            (Tensor): Transformed latent tensor of shape [LATENT_DIM x BATCH_SIZE x EMBED_DIM].
    """

    def __init__(self, embed_dim, mlp_dim, n_heads, dropout, n_layers):
        super().__init__()

        # Create a list of decoder layers (PerceiverAttention blocks)
        self.transformer = nn.ModuleList([])
        
        for _ in range(n_layers):
            self.transformer.append(
                PerceiverAttentionBlock(
                    embed_dim=embed_dim,
                    mlp_dim=mlp_dim,
                    n_heads=n_heads,
                    dropout=dropout)
                )

    def forward(self, latent):
        """
        Forward pass of the LatentTransformer module.

        Args:
            latent (Tensor): Latent tensor of shape [LATENT_DIM x BATCH_SIZE x EMBED_DIM].

        Returns:
            (Tensor): Transformed latent tensor of shape [LATENT_DIM x BATCH_SIZE x EMBED_DIM].

        """
        for transform in self.transformer:
            latent = transform(latent, latent)
        return latent

class PerceiverBlock(nn.Module):
    """
    Block consisting of one cross-attention layer and one latent transformer.

    Args:
        embed_dim (int): Dimension of the embedded representations.
        attention_mlp_dim (int): Dimension of the cross-attention's feedforward network hidden layer.
        transformer_mlp_dim (int): Dimension of the latent transformer's feedforward network hidden layer.
        transformer_heads (int): Number of attention heads for the latent transformer.
        dropout (float): Dropout probability.
        transformer_layers (int): Number of layers in the latent transformer.

    """
    def __init__(self, embed_dim, attention_mlp_dim, transformer_mlp_dim, transformer_heads, dropout, transformer_layers):
        super().__init__()
        
        # Cross-Attention layer
        self.cross_attention = PerceiverAttentionBlock(
            embed_dim=embed_dim, 
            mlp_dim=attention_mlp_dim, 
            n_heads=1, 
            dropout=dropout)

        # Latent Transformer module
        self.latent_transformer = LatentTransformer(
            embed_dim=embed_dim, 
            mlp_dim=transformer_mlp_dim, 
            n_heads=transformer_heads, 
            dropout=dropout, 
            n_layers=transformer_layers)

    def forward(self, x, latent):
        """
        Forward pass of the PerceiverBlock module.

        Args:
            x (Tensor): Input tensor of shape [PIXELS x BATCH_SIZE x EMBED_DIM].
            latent (Tensor): Latent tensor of shape [LATENT_DIM x BATCH_SIZE x EMBED_DIM].

        Returns:
            (Tensor): Transformed latent tensor of shape [LATENT_DIM x BATCH_SIZE x EMBED_DIM].
        """
        
        # Apply cross-attention on the input and latent tensor
        latent = self.cross_attention(x, latent)

        # Apply the latent transformer
        latent = self.latent_transformer(latent)

        return latent

class Classifier(nn.Module):
    """
    Classifier for Perceiver model for binary classification (AD or NC)

    Args:
        embed_dim (int): Dimension of the embedded representations.
        n_classes (int): Number of target classes. Default to 2 to classify AD or NC
        
    Returns:
        (Tensor): Output tensor for classification of shape [n_classes].
    """

    def __init__(self, embed_dim, n_classes=2):
        super().__init__()

        # First fully connected layer
        self.fc1 = nn.Linear(embed_dim, embed_dim)

        # Second fully connected layer for classification
        self.fc2 = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        """
        Forward pass of the Classifier module.

        Args:
            x (Tensor): Input tensor of shape [LATENT_DIM x BATCH_SIZE x EMBED_DIM].

        Returns:
            Tensor: Output tensor for classification of shape [n_classes].

        """

        # Apply the first fully connected layer and take the mean along the batch dimension
        x = self.fc1(x).mean(dim=0)

        # Apply the second fully connected layer for classification
        x = self.fc2(x)

        return x
    

class Perceiver(nn.Module):
    """
    Perceiver Model for Alzheimer's Disease Classification

    Args:
        input_shape (tuple): Shape of the input images (e.g., (1, 256, 256)).
        latent_dim (int): Dimension of the latent vectors.
        embed_dim (int): Dimension of the embedded representations.
        attention_mlp_dim (int): Dimension of the attention module hidden layer.
        transformer_mlp_dim (int): Dimension of the transformer block hidden layer.
        transformer_heads (int): Number of attention heads in the transformer.
        dropout (float): Dropout probability.
        transformer_layers (int): Number of transformer layers in the Perceiver.
        n_blocks (int): Number of Perceiver blocks.
        n_classes (int): Number of target classes.
        batch_size (int): Batch size.
        
    Returns:
        (Tensor): Output tensor for classification of shape [n_classes].
    """

    def __init__(
        self, input_shape, latent_dim, embed_dim, attention_mlp_dim, transformer_mlp_dim, transformer_heads,
        dropout, transformer_layers, n_blocks, n_classes):
        
        super().__init__()
        # Initialise the latent array
        self.latent = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.zeros((latent_dim, 1, embed_dim)), mean=0, std=0.02, a=-2, b=2
            )
        )

        # Initialise the image embedding with position encoding
        self.embed = PositionalImageEmbedding(input_shape=input_shape, input_channels=3, embed_dim=embed_dim)

        # Initialize multiple Perceiver blocks
        self.perceiver_blocks = nn.ModuleList([])
        
        for _ in range(n_blocks):
            self.perceiver_blocks.append(
                PerceiverBlock(
                    embed_dim=embed_dim,
                    attention_mlp_dim=attention_mlp_dim,
                    transformer_mlp_dim=transformer_mlp_dim,
                    transformer_heads=transformer_heads,
                    dropout=dropout,
                    transformer_layers=transformer_layers)
                )

        # Initialise the binary classification layer for Alzheimer's disease
        self.classifier = Classifier(embed_dim=embed_dim, n_classes=n_classes)

    def forward(self, x):
        """
        Forward pass of the Perceiver model.
        
        Args:
            x (Tensor): Input tensor of shape [BATCH_SIZE x CHANNELS x HEIGHT x WIDTH].
        
        Returns:
            (Tensor): Output tensor for classification of shape [n_classes].
        """
        # Expand the latent query matrix to match the batch size
        batch_size = x.shape[0]
        latent = self.latent.expand(-1, batch_size, -1)

        # Pass the input image through the embedding module for feature extraction
        x = self.embed(x)

        # Iteratively pass the latent matrix and image embedding through Perceiver blocks
        for perceiver_block in self.perceiver_blocks:
            latent = perceiver_block(x, latent)

        # Perform binary classification to distinguish Alzheimer's disease
        # from non-Alzheimer's
        latent = self.classifier(latent)

        return latent
    
    
def create_model(input_shape, latent_dim, embed_dim, attention_mlp_dim, transformer_mlp_dim, transformer_heads, dropout, transformer_layers, n_blocks, n_classes, lr):
    """
    Create the Perceiver model for Alzheimer's Disease classification.
    
    Args:
        input_shape (tuple): Shape of the input images (e.g., (1, 256, 256)).
        latent_dim (int): Dimension of the latent vectors.
        embed_dim (int): Dimension of the embedded representations.
        attention_mlp_dim (int): Dimension of the attention module hidden layer.
        transformer_mlp_dim (int): Dimension of the transformer block hidden layer.
        transformer_heads (int): Number of attention heads in the transformer.
        dropout (float): Dropout probability.
        transformer_layers (int): Number of transformer layers in the Perceiver.
        n_blocks (int): Number of Perceiver blocks.
        n_classes (int): Number of target classes.
        lr (float): Learning rate.
        
    Returns:
        model (Perceiver): Perceiver model for Alzheimer's Disease classification.
        optimizer (torch.optim.Adam): Adam optimizer.
        criterion (torch.nn.CrossEntropyLoss): Cross-entropy loss function.
        scheduler (torch.optim.lr_scheduler.StepLR): Learning rate scheduler.
    """
    model = Perceiver(input_shape, latent_dim, embed_dim, attention_mlp_dim, transformer_mlp_dim, transformer_heads, dropout, transformer_layers, n_blocks, n_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=1,gamma=0.8)
    return model, optimizer, criterion, scheduler
