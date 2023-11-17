""" Source of the components of model. """

# Acknowledgement: This code is adapted from the source found in reference [6]

import torch
from torch import nn

class PatchEmbedding(nn.Module):
    """ Turns a 2D input image into a 1D sequence learnable embedding vector. """ 
    def __init__(self, 
                in_channels,        # Number of input image channels
                patch_size,         # Patch size (assume square patches)
                embedding_dim):     # Embedding dimension for tokens
        super().__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        # Create a layer to turn an image into patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                out_channels=embedding_dim,
                                kernel_size=patch_size,
                                stride=patch_size,
                                padding=0)

        # Create a layer to flatten the patch feature maps into a single dimension
        self.flattened = nn.Flatten(start_dim=2,
                                    end_dim=3)

    def forward(self, x):
        x_patched = self.patcher(x)
        x_flattened = self.flattened(x_patched)

        # Make sure the output shape has the right order 
        # adjust so the embedding is on the final dimension
        return x_flattened.permute(0, 2, 1)

class MSABlock(nn.Module):
    """ Creates a multi-head self-attention block. """
    def __init__(self,
                embedding_dim,  # Hidden size D
                num_heads,      # Number of heads
                attn_dropout): 
        super().__init__()

        # Create the Normalisation layer
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Create the MSA layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True)

    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x,   # query embeddings 
                                            key=x,      # key embeddings
                                            value=x,    # value embeddings
                                            need_weights=False)
        return attn_output

class MLPBlock(nn.Module):
    """ Creates a layer normalized multilayer perceptron block. """
    def __init__(self,
                embedding_dim,  # Hidden Size D
                mlp_size,       # MLP size
                dropout):       # Dropout
        super().__init__()

        # Create the Normalisation layer
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Create the MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),
            nn.Dropout(p=dropout) 
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x

class TransformerEncoderBlock(nn.Module):
    """ Creates a Transformer Encoder block. """
    def __init__(self,
                embedding_dim,  # Hidden size D
                num_heads,      # Number of heads
                mlp_size,       # MLP size
                mlp_dropout,    # Amount of dropout for dense layers
                attn_dropout):  # Amount of dropout for attention layers
        super().__init__()

        # Create MSA block
        self.msa_block = MSABlock(embedding_dim=embedding_dim,
                                    num_heads=num_heads,
                                    attn_dropout=attn_dropout)

        # Create MLP block
        self.mlp_block =  MLPBlock(embedding_dim=embedding_dim,
                                mlp_size=mlp_size,
                                dropout=mlp_dropout)

    def forward(self, x):
        # Create residual connection for MSA block
        x =  self.msa_block(x) + x 

        # Create residual connection for MLP block
        x = self.mlp_block(x) + x 

        return x

class ViT(nn.Module):
    """ Creates a Vision Transformer architecture with ViT-Base hyperparameters by default. """
    def __init__(self,
                img_size,               # Image resolution
                in_channels,            # Number of channels in input image
                patch_size,             # Patch size
                num_transformer_layers, # LNumber of transformer layers
                embedding_dim,          # Hidden size D
                mlp_size,               # MLP size
                num_heads,              # Number of heads
                attn_dropout,           # Dropout for attention projection
                mlp_dropout,            # Dropout for dense/MLP layers 
                embedding_dropout,      # Dropout for patch and position embeddings
                num_classes):           # Number of classes
        super().__init__()

        # Calculate number of patches
        self.num_patches = (img_size * img_size) // patch_size**2

        # Create learnable class embedding (needs to go at front of sequence of patch embeddings)
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)
        
        # Create learnable position embedding
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim),
                                            requires_grad=True)

        # Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        
        # Create patch embedding layer
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                            patch_size=patch_size,
                                            embedding_dim=embedding_dim)
        
        # Create Transformer Encoder block
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                            num_heads=num_heads,
                                                                            mlp_size=mlp_size,
                                                                            mlp_dropout=mlp_dropout,
                                                                            attn_dropout=attn_dropout) for _ in range(num_transformer_layers)])

        # Create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, 
                    out_features=num_classes)
        )

    def forward(self, x):

        # Get batch size
        batch_size = x.shape[0]

        # Create class token embedding and expand it to match the batch size
        class_token = self.class_embedding.expand(batch_size, -1, -1)

        # Create patch embedding
        x = self.patch_embedding(x)

        # Concat class embedding and patch embedding
        x = torch.cat((class_token, x), dim=1)

        # Add position embedding to patch embedding
        x = self.position_embedding + x

        # Run embedding dropout (Appendix B.1)
        x = self.embedding_dropout(x)

        # Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
        x = self.transformer_encoder(x)

        # Put 0 index logit through classifier
        x = self.classifier(x[:, 0])

        return x