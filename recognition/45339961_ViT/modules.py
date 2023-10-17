""" Source of the components of model. """
# Acknowledgement: This code is adapted from the following source:
# https://www.learnpytorch.io/08_pytorch_paper_replicating/#44-flattening-the-patch-embedding-with-torchnnflatten

import torch
from torch import nn

class PatchEmbedding(nn.Module):
    """ Turns a 2D input image into a 1D sequence learnable embedding vector. """ 
    def __init__(self, 
                in_channels=1,          # Number of input image channels (1 for ADNI default)
                patch_size=16,          # Patch size (assume square patches), default is ViT paper
                embedding_dim=768):     # Embedding dimension for tokens (number of channels created from each patch
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
        self.flattened = nn.Flatten(start_dim=2, # only flatten the feature map dimensions into a single vector
                                end_dim=3)

    def forward(self, x):
        x_patched = self.patcher(x)
        x_flattened = self.flattened(x_patched)

        # Make sure the output shape has the right order 
        # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]
        return x_flattened.permute(0, 2, 1)

class MSABlock(nn.Module):
    """ Creates a multi-head self-attention block. """
    def __init__(self,
                embedding_dim=768,  # Hidden size D from Table 1 for ViT-Base
                num_heads=12,       # Heads from Table 1 for ViT-Base
                attn_dropout=0.0):  # Paper doesn't use any dropout in MSABlocks
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
                embedding_dim=768,  # Hidden Size D from Table 1 for ViT-Base
                mlp_size=3072,      # MLP size from Table 1 for ViT-Base
                dropout=0.1):       # Dropout from Table 3 for ViT-Base
        super().__init__()

        # Create the Normalisation layer
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Create the MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                    out_features=mlp_size),
            nn.GELU(),                              # "The MLP contains two layers with a GELU non-linearity (section 3.1)."
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,         # needs to take same in_features as out_features of layer above
                    out_features=embedding_dim),    # take back to embedding_dim
            nn.Dropout(p=dropout)                   # "Dropout, when used, is applied after every dense layer.."
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x

class TransformerEncoderBlock(nn.Module):
    """ Creates a Transformer Encoder block. """
    def __init__(self,
                embedding_dim=768,  # Hidden size D from Table 1 for ViT-Base
                num_heads=12,       # Heads from Table 1 for ViT-Base
                mlp_size=3072,      # MLP size from Table 1 for ViT-Base
                mlp_dropout=0.1,    # Amount of dropout for dense layers from Table 3 for ViT-Base
                attn_dropout=0.0):  # Amount of dropout for attention layers
        super().__init__()

        # Create MSA block (equation 2)
        self.msa_block = MSABlock(embedding_dim=embedding_dim,
                                    num_heads=num_heads,
                                    attn_dropout=attn_dropout)

        # Create MLP block (equation 3)
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
                img_size=224,               # Training resolution from Table 3 in ViT paper
                in_channels=3,              # Number of channels in input image
                patch_size=16,              # Patch size
                num_transformer_layers=12,  # Layers from Table 1 for ViT-Base
                embedding_dim=768,          # Hidden size D from Table 1 for ViT-Base
                mlp_size=3072,              # MLP size from Table 1 for ViT-Base
                num_heads=12,               # Heads from Table 1 for ViT-Base
                attn_dropout=0.0,           # Dropout for attention projection
                mlp_dropout=0.1,            # Dropout for dense/MLP layers 
                embedding_dropout=0.1,      # Dropout for patch and position embeddings
                num_classes=2):             # 2 classes in ADNI dataset
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
        
        # Create Transformer Encoder blocks (can stack Transformer Encoder blocks using nn.Sequential()) 
        # Note: The "*" means "all"
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                            num_heads=num_heads,
                                                                            mlp_size=mlp_size,
                                                                            mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])

        # Create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, 
                    out_features=num_classes)
        )

    def forward(self, x):

        # Get batch size
        batch_size = x.shape[0]

        # Create class token embedding and expand it to match the batch size (equation 1)
        class_token = self.class_embedding.expand(batch_size, -1, -1)

        # Create patch embedding (equation 1)
        x = self.patch_embedding(x)

        # Concat class embedding and patch embedding (equation 1)
        x = torch.cat((class_token, x), dim=1)

        # Add position embedding to patch embedding (equation 1) 
        x = self.position_embedding + x

        # Run embedding dropout (Appendix B.1)
        x = self.embedding_dropout(x)

        # Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
        x = self.transformer_encoder(x)

        # Put 0 index logit through classifier (equation 4)
        x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index

        return x