""" Source of the components of model. """

import torch
from torch import nn

# class PatchEmbeddingLayer(nn.Module):
#     """ Creates a patch embedding layer """
#     def __init__(self, 
#                 in_channels, 
#                 patch_size, 
#                 embedding_dim,
#                 batch_size,
#                 n_patches
#             ):
#         super().__init__()

#         # Attributes
#         self.patch_size = patch_size
#         self.embedding_dim = embedding_dim
#         self.in_channels = in_channels

#         # Convolutional embedding layer
#         self.conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size)

#         # Flatten layer
#         self.flatten_layer = nn.Flatten(start_dim=1, end_dim=2)

#         # Class token
#         self.class_token_embeddings = nn.Parameter(torch.rand((batch_size, 1, embedding_dim), requires_grad=True))

#         # Positional embeddings
#         self.position_embeddings = nn.Parameter(torch.rand((1, n_patches + 1, embedding_dim), requires_grad=True))

#     def forward(self, x):
#         # Extract patches and apply convolution
#         patches = self.conv_layer(x).permute((0, 2, 3, 1))

#         # Flatten patches
#         flattened_patches = self.flatten_layer(patches)

#         # Concatenate class token to flattened patches
#         patch_embeddings = torch.cat((self.class_token_embeddings, flattened_patches), dim=1)

#         # Add positional embeddings
#         output =  patch_embeddings + self.position_embeddings

#         return output

# class MSABlock(nn.Module):
#     """ Creates a layer normalised multi-head self attention block """
#     def __init__(self,
#                 embedding_dims, # Hidden Size D in the ViT Paper Table 1
#                 num_heads,  # Heads in the ViT Paper Table 1
#                 attn_dropout=0 # Default to Zero as there is no dropout for the the MSA Block as per the ViT Paper
#             ):
#         super().__init__()

#         # Attributes
#         self.embedding_dims = embedding_dims
#         self.num_head = num_heads
#         self.attn_dropout = attn_dropout

#         # Layer Normalisation Layer
#         self.layernorm = nn.LayerNorm(normalized_shape = embedding_dims)

#         # Multihead Attention Layer
#         self.multiheadattention =  nn.MultiheadAttention(num_heads = num_heads,
#                                                         embed_dim = embedding_dims,
#                                                         dropout = attn_dropout,
#                                                         batch_first = True
#                                                         )

#     def forward(self, x):
#         # Normalisation
#         x = self.layernorm(x)

#         # Self Attention
#         output,_ = self.multiheadattention(query=x,
#                                             key=x,
#                                             value=x,
#                                             need_weights=False)
#         return output

# class MLPBlock(nn.Module):
#     """ Creates a layer normalisaed multi-layer perceptron block """
#     def __init__(self, 
#                 embedding_dims,     # Hidden Size D in the ViT Paper Table 1
#                 mlp_size,           # MLP Size in the ViT Paper Table 1
#                 mlp_dropout         # Dropout in the ViT Paper Table 3
#             ):
#         super().__init__()

#         # Attributes
#         self.embedding_dims = embedding_dims
#         self.mlp_size = mlp_size
#         self.dropout = mlp_dropout

#         # Normalisation Layer (LN)
#         self.layernorm = nn.LayerNorm(normalized_shape = embedding_dims)

#         # Create MLP Layer
#         self.mlp = nn.Sequential(
#             nn.Linear(in_features = embedding_dims,
#                         out_features = mlp_size),
#             nn.GELU(),
#             nn.Dropout(p = mlp_dropout),
#             nn.Linear(in_features = mlp_size,       # Needs to take same in_features as out_features from above layer
#                     out_features = embedding_dims), # Take back to embedding dimensions
#             nn.Dropout(p = mlp_dropout)             # Dropout, when used, is applied after every dense layer
#         )

#     def forward(self, x):
#         normalised_layer = self.layernorm(x)
#         return self.mlp(normalised_layer)

# class TransformerEncoderBlock(nn.Module):
#     """ Creates a transformer encoder block """
#     def __init__(self, 
#                 embedding_dims, # Hidden Size D in the ViT Paper Table 1
#                 mlp_dropout,    # Dropout for dense layers in the ViT Paper Table 3
#                 attn_dropout,   # Dropout for attention layers in the ViT Paper Table 3
#                 mlp_size,       # MLP Size in the ViT Paper Table 1
#                 num_heads,      # Heads in the ViT Paper Table 1
#             ):
#         super().__init__()

#         # Create MSA block (Equation 2)
#         self.msa_block = MSABlock(embedding_dims = embedding_dims,
#                                                     num_heads = num_heads,
#                                                     attn_dropout = attn_dropout)

#         # Create MLP block (Equation 3)
#         self.mlp_block = MLPBlock(embedding_dims = embedding_dims,
#                                                         mlp_size = mlp_size,
#                                                         mlp_dropout = mlp_dropout)

#     def forward(self,x):
#         # Create residual connection for MSA block
#         x = self.msa_block(x) + x

#         # Create residual connection for MLP block
#         x = self.mlp_block(x) + x

#         return x

# class ViT(nn.Module):
#     def __init__(self,
#                 img_size,
#                 in_channels,
#                 patch_size,
#                 embedding_dims,
#                 num_transformer_layers,
#                 mlp_dropout,
#                 attn_dropout,
#                 embedding_dropout,
#                 mlp_size,
#                 num_heads,
#                 num_classes,
#                 batch_size):
#         super().__init__()

#         self.n_patches = (img_size * img_size) // patch_size**2

#         self.patch_embedding_layer = PatchEmbeddingLayer(in_channels=in_channels, 
#                                                         patch_size=patch_size, 
#                                                         embedding_dim=embedding_dims,
#                                                         batch_size=batch_size,
#                                                         n_patches=self.n_patches)

#         self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dims = embedding_dims,
#                                                 mlp_dropout = mlp_dropout,
#                                                 attn_dropout = attn_dropout,
#                                                 mlp_size = mlp_size,
#                                                 num_heads = num_heads) for _ in range(num_transformer_layers)])

#         self.classifier = nn.Sequential(nn.LayerNorm(normalized_shape = embedding_dims),
#                                         nn.Linear(in_features = embedding_dims,
#                                                 out_features = num_classes))

#     def forward(self, x):
#         return self.classifier(self.transformer_encoder(self.patch_embedding_layer(x))[:, 0])

class PatchEmbedding(nn.Module):
    """ Turns a 2D input image into a 1D sequence learnable embedding vector. """ 
    def __init__(self, 
                in_channels=3,          # Number of input image channels
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
        self.flatten = nn.Flatten(start_dim=2, # only flatten the feature map dimensions into a single vector
                                end_dim=3)

    def forward(self, x):
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)

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
        attn_output, _ = self.multihead_attn(query=x, # query embeddings 
                                            key=x, # key embeddings
                                            value=x, # value embeddings
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
    # 2. Initialize the class with hyperparameters from Table 1 and Table 3
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