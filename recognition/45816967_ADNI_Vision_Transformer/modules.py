import torch
import torch.nn as nn

patch_size = 16
height = 224
width = 224

# Patch Embedding Module Modified from: https://www.learnpytorch.io/08_pytorch_paper_replicating/#44-flattening-the-patch-embedding-with-torchnnflatten
class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.
    
    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 1.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 256.
    """ 
    def __init__(self, 
                 in_channels:int=1,
                 patch_size:int=16,
                 embedding_dim:int=256):
        super().__init__()
        
        self.patch_size = patch_size
        
        # Layer which converts images into patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        # Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2,
                                  end_dim=3)

    def forward(self, x):
        # Get Image Shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {patch_size}"
        
        # Split into patches
        x_patched = self.patcher(x)
        
        # Flatten patches
        x_flattened = self.flatten(x_patched) 
        
        # Permute to make embedding dimension the last dimension
        return x_flattened.permute(0, 2, 1)

# Vision Transformer Model Modified from: https://www.learnpytorch.io/08_pytorch_paper_replicating/#44-flattening-the-patch-embedding-with-torchnnflatten
class ViT(nn.Module):
    def __init__(self,
                 img_size:int=224,
                 in_channels:int=1,
                 patch_size:int=16,
                 num_transformer_layers:int=8,
                 embedding_dim:int=128,
                 mlp_size:int=128,
                 num_heads:int=8,
                 dropout:float=0.1,
                 embedding_dropout:float=0.1,
                 num_classes:int=2):
        super().__init__()
        
        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."
        
        # Calculate number of patches (height * width/patch^2)
        self.num_patches = (img_size * img_size) // patch_size**2
                 
        # Create learnable class embedding
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
        # Transformer Encoder layers made up of MHSA and MLP blocks
        self.transformer_encoder = nn.Sequential(*[nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                             nhead=num_heads,
                                                             dim_feedforward=mlp_size,
                                                             dropout=dropout,
                                                             activation="gelu",
                                                             batch_first=True,
                                                             norm_first=True) for _ in range(num_transformer_layers)])
       
        
        # Create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, 
                      out_features=num_classes),
            nn.Softmax(dim=-1) # calculate softmax across the last dimension
        )
    
    def forward(self, x):
        
        # Get batch size
        batch_size = x.shape[0]
        
        # Create class token embedding and expand it to match the batch size (equation 1)
        class_token = self.class_embedding.expand(batch_size, -1, -1) # "-1" means to infer the dimension

        # Create patch embedding
        x = self.patch_embedding(x)

        # Concat class embedding and patch embedding
        x = torch.cat((class_token, x), dim=1)

        # Add position embedding to patch embedding
        x = self.position_embedding + x

        # Run embedding dropout
        x = self.embedding_dropout(x)

        # Pass patch, position and class embedding through transformer encoder layers
        x = self.transformer_encoder(x)

        # Put 0 index logit through classifier
        x = self.classifier(x[:, 0])

        return x       