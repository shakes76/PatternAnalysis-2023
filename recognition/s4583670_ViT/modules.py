'''
@file   modules.py
@brief  Contains the source code of all the components of the vision transformer model.
        It defines the InputEmbedding object, Encoder object and ViT Object
@date   20/10/2023
'''

import torch
import torch.nn as nn

'''
InputEmbedding

This class defines the input embedding module of the ViT. The object will take in the batch input,
spit the images into patches using a convolutional, project these patches onto a linear plane, prepend
the class embedded token and finally add the positional embedding tokens. The output of this model is the 
embedding tokens to pass into the transformer encoder.
'''
class InputEmbedding(nn.Module):
    def __init__(self, args) -> None:
        super(InputEmbedding, self).__init__()
        self.batch_size = args.batch_size
        self.mlp_dim = args.mlp_dim
        self.head_dim = args.head_dim
        self.n_channels = args.n_channels
        self.patch_size = args.patch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = args.hidden_size

        # Convolutional layer to patchify images
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=args.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding='valid'
        )

        # Positional embeddings
        self.positionalEmbedding = nn.Parameter(torch.randn(self.batch_size, 1, self.mlp_dim)).to(self.device)

        # Class token
        self.classToken = nn.Parameter(torch.randn(self.batch_size, 1, self.mlp_dim)).to(self.device)

        # Linear projection layer
        self.linearProjection = nn.Linear(self.input_size, self.mlp_dim)

    def forward(self, input):
        input = input.to(self.device)

        # Patch the image using a convolutional layer
        patches = self.conv1(input)
        seq_len = (input.shape[2] // self.patch_size) * (input.shape[3] // self.patch_size)
        imagePatches = torch.reshape(patches, [-1, seq_len, self.input_size])

        # Project the patched images onto a linear plane using a FC linear layer
        linearProjection = self.linearProjection(imagePatches).to(self.device)

        # Define the class token
        self.classToken = nn.Parameter(torch.randn(linearProjection.shape[0], 1, self.mlp_dim)).to(self.device)

        # Concatenate the class token to the embedding tokens
        linearProjection = torch.cat((self.classToken, linearProjection), dim=1)

        # Add the positional embeddings to the input embeddings and class token
        linearProjection += self.positionalEmbedding[:linearProjection.shape[0], :linearProjection.shape[1] + 1, :]
        return linearProjection
    
'''
Encoder

This class defines the encoder block for the ViT. It implements the transformer encoder architecture as described in the
README.md. The encoder takes the embedding tokens as an input, and passes these through the transformer.
'''
class Encoder(nn.Module):
    def __init__(self, args) -> None:
        super(Encoder, self).__init__()
        self.dropout = args.dropout
        self.num_heads = args.num_heads
        self.mlp_dim = args.mlp_dim
        self.head_dim = args.head_dim

        # Normalisation layer
        self.normLayer = nn.LayerNorm(self.mlp_dim)

        # Multi-head Attention Layer
        self.attention = nn.MultiheadAttention(self.mlp_dim, self.num_heads, dropout=self.dropout)

        # MLP Encoder
        self.encoderMLP = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
    
    def forward(self, embeddedPatches):
        # Normalise the embedded patches
        normalisation = self.normLayer(embeddedPatches)

        # Multi head attention output
        attentionOut = self.attention(normalisation, normalisation, normalisation)[0]

        # Second normalisation block
        normalisation = self.normLayer(attentionOut + embeddedPatches)

        # Encoder output
        return (self.encoderMLP(normalisation) + attentionOut + embeddedPatches)
    
'''
ViT

This class defines the vision transformer architecture. It contains the code to interface between
the modules of the vision transformer (i.e. input embedding object to encoder objects to MLP head for 
classification). The object takes a batch of images as the input.
'''   
class ViT(nn.Module):
    def __init__(self, args) -> None:
        super(ViT, self).__init__()
        self.dropout = args.dropout
        self.num_classes = args.num_classes
        self.num_encoders = args.num_encoders
        self.mlp_dim = args.mlp_dim
        self.head_dim = args.head_dim

        # Transformer encoder layer(s)
        self.encoders = nn.ModuleList([Encoder(args) for i in range(self.num_encoders)])

        # Input embedding layers
        self.embedding = InputEmbedding(args)

        # MLP head for classification
        self.MLP = nn.Sequential(
            nn.LayerNorm(self.mlp_dim),
            nn.Linear(self.mlp_dim, self.head_dim),
            nn.Linear(self.head_dim, self.num_classes)
        )

    def forward(self, input):
        # Get the embedding of the input
        encoderOut = self.embedding(input)

        # Loop through all the encoder blocks
        for layer in self.encoders:
            encoderOut = layer(encoderOut)

        # Output of MLP head is classification resul
        out = self.MLP(torch.mean(encoderOut, dim=1))
        return out