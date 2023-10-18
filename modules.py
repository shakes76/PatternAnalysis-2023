import torch
import torch.nn as nn

batch_size = 128
workers = 1

# Images are 256 by 240 pixels. Resize them to 224 by 224; must be divisible by 16
image_size = 224  # Resized 2D image input
patch_size = 16  # Dimension of a patch
num_patches = (image_size // patch_size) ** 2  # Number of patches in total
num_channels = 3  # 3 channels for RGB
embed_dim = 768  # Hidden size D of ViT-Base model from paper, equal to [(patch_size ** 2) * num_channels]
num_heads = 12  # Number of self attention blocks
num_layers = 12  # Number of Transformer encoder layers
mlp_size = 3072  # Number of hidden units between each linear layer
dropout_size = 0.1
num_classes = 2  # Number of different classes to classify (i.e. AD and NC)
num_epochs = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not found. Using CPU")

# ------------------------------------------------------------------
# Patch Embedding
class PatchEmbedding(nn.Module):
    """Takes a 2D input image and splits it into fixed-sized patches and linearly embeds each of them.

    Changes the dimensions from H x W x C to N x (P^2 * C), where 
    (H, W, C) is the height, width, number of channels of the image,
    N is the number of patches, 
    and P is the dimension of each patch; P^2 represents a flattened patch.
    """
    def __init__(self, ngpu):
        super(PatchEmbedding, self).__init__()
        self.ngpu = ngpu
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
    def __init__(self, ngpu):
        super(TransformerEncoder, self).__init__()
        self.ngpu = ngpu

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
    """
    def __init__(self, ngpu):
        super(MLPHead, self).__init__()
        self.ngpu = ngpu

        self.main = nn.Sequential(nn.LayerNorm(normalized_shape=embed_dim),
                                    nn.Linear(in_features=embed_dim,
                                                out_features=num_classes)
                                    )
    
    def forward(self, input):
        return self.main(input)

# ------------------------------------------------------------------
# Visual Transformer
class ViT(nn.Module):
    """Creates a vision transformer model."""
    def __init__(self, ngpu):
        super(ViT, self).__init__()
        self.ngpu = ngpu

        self.patch_embedding = PatchEmbedding(workers)
        self.prepend_embed_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)
        self.position_embed_token = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim), requires_grad=True)
        self.embedding_dropout = nn.Dropout(p=dropout_size)  # Apply dropout after positional embedding as well
        self.transformer_encoder = TransformerEncoder(workers)
        self.mlp_head = MLPHead(workers)

    def forward(self, input):
        current_batch_size = input.size(0)
        prepend_embed_token_expanded = self.prepend_embed_token.expand(current_batch_size, -1, -1)
        input = self.patch_embedding(input)  # Patch embedding
        input = torch.cat((prepend_embed_token_expanded, input), dim=1)  # Prepend class token
        input = input + self.position_embed_token  # Add position embedding
        input = self.embedding_dropout(input)  # Apply dropout
        input = self.transformer_encoder(input)  # Feed into transformer encoder layers
        input = self.mlp_head(input[:, 0])  # Get final classificaiton from MLP head
        return input
