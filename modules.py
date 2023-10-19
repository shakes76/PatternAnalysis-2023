import torch
import torch.nn as nn

# ------------------------------------------------------------------
# Patch Embedding
class PatchEmbedding(nn.Module):
    """Takes a 2D input image and splits it into fixed-sized patches and linearly embeds each of them.

    Changes the dimensions from H x W x C to N x (P^2 * C), where 
    (H, W, C) is the height, width, number of channels of the image,
    N is the number of patches, 
    and P is the dimension of each patch; P^2 represents a flattened patch.
    """
    def __init__(self, num_channels: int, embed_dim: int, patch_size: int):
        super(PatchEmbedding, self).__init__()
        # Puts image through Conv2D layer with kernel_size = stride to ensure no patches overlap.
        # This will split image into fixed-sized patches; each patch has the same dimensions
        # Then, each patch is flattened, including all channels for each patch.
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=num_channels,
                        out_channels=embed_dim,
                        kernel_size=patch_size,
                        stride=patch_size,
                        padding=0),
            nn.Flatten(start_dim=2, end_dim=3)
        )

    def forward(self, input):
        return self.main(input).permute(0, 2, 1)  # Reorder the dimensions
    
# ------------------------------------------------------------------
# Transformer Encoder
class TransformerEncoder(nn.Module):
    """Creates a standard Transformer Encoder.
    One transformer encoder layer consists of layer normalisation, multi-head self attention layer, 
    a residual connection, another layer normalisation, an mlp block, and another residual connection.
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_size: int, dropout_size: float, num_layers: int):
        super(TransformerEncoder, self).__init__()

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                                    nhead=num_heads,
                                                                    dim_feedforward=mlp_size,
                                                                    dropout=dropout_size,
                                                                    activation="gelu",
                                                                    layer_norm_eps=1e-5,
                                                                    batch_first=True,
                                                                    norm_first=True
                                                                    )
        
        self.full_transformer_encoder = nn.TransformerEncoder(encoder_layer=self.transformer_encoder_layer,
                                                                num_layers=num_layers)
        
    
    def forward(self, input):
        return self.full_transformer_encoder(input)

# ------------------------------------------------------------------
# MLP head
class MLPHead(nn.Module):
    """Creates an MLP head.
    Consists of a layer normalisation and a linear layer.
    Used for classficiation.
    """
    def __init__(self, embed_dim: int, num_classes: int):
        super(MLPHead, self).__init__()
        self.main = nn.Sequential(nn.LayerNorm(normalized_shape=embed_dim),
                                    nn.Linear(in_features=embed_dim,
                                                out_features=num_classes)
                                    )
    
    def forward(self, input):
        return self.main(input)

# ------------------------------------------------------------------
# Visual Transformer
class ViT(nn.Module):
    """Creates a vision transformer model.
    Contains patch embedding, position embedding, transformer encoder layers, and MLP head for classification.
    Can initiate with default values for ADNI dataset.
    """
    def __init__(self, num_channels: int = 1, embed_dim: int = 768, patch_size: int = 16, num_heads: int = 12, 
                    mlp_size: int = 3072, dropout_size: float = 0.1, num_layers: int = 12, num_classes: int = 2,
                    image_size: int = 224):
        super(ViT, self).__init__()
        
        # Initiate patch embedding
        self.patch_embedding = PatchEmbedding(num_channels=num_channels, embed_dim=embed_dim, patch_size=patch_size)

        # Prepare the class token
        self.prepend_embed_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)

        # Prepare the position embedding
        num_patches = (image_size // patch_size) ** 2
        self.position_embed_token = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim), requires_grad=True)

        # Apply dropout after positional embedding
        self.embedding_dropout = nn.Dropout(p=dropout_size)

        # Initiate TransformerEncoder layers
        self.transformer_encoder = TransformerEncoder(embed_dim=embed_dim, num_heads=num_heads, mlp_size=mlp_size, 
                                                      dropout_size=dropout_size, num_layers=num_layers)
        
        # Initiate MLPHead
        self.mlp_head = MLPHead(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, input):
        current_batch_size = input.size(0)

        # Expand the class token to batch size
        prepend_embed_token_expanded = self.prepend_embed_token.expand(current_batch_size, -1, -1)

        # Apply Patch embedding
        input = self.patch_embedding(input) 

        # Prepend class token
        input = torch.cat((prepend_embed_token_expanded, input), dim=1)  

        # Add position embedding
        input = input + self.position_embed_token  

        # Apply dropout
        input = self.embedding_dropout(input)  

        # Feed into transformer encoder layers
        input = self.transformer_encoder(input)  

        # Get final classificaiton from MLP head
        input = self.mlp_head(input[:, 0])  
        return input
