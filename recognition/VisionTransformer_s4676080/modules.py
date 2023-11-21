"""
modules.py: This module defines the neural network components utilized in the Vision Transformer architecture.
It provides classes for patch embedding and the main Vision Transformer model suitable for image classification tasks.
"""

# importing libraries 
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Patch Embedding layer.    
    This layer uses a convolutional operation to transform patches of the input image into
    vectors that can be processed by the subsequent transformer layers.
    
    Parameters:
    - in_channels: Number of input channels.
    - patch_size: Size of each square patch.
    - emb_size: Dimension of embedding vectors.
    """
    def __init__(self, in_channels=3, patch_size=16, emb_size=768):        
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(emb_size)
        )

    def forward(self, x):
        """
        Forward pass for the PatchEmbedding layer.
        
        Parameters:
        - x: Input tensor with shape (batch_size, in_channels, height, width).
        
        Returns:
            Embedded tensor with shape
        """
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  
        return x
    
class VisionTransformer(nn.Module):
    """
    The Vision Transformer model that uses patch embeddings and a sequence of transformer 
    blocks to process the input image for classification tasks.
    
    Parameters:
    - in_channels: Number of input channels.
    - patch_size: Size of each square patch.
    - emb_size: Dimension of embedding vectors.
    - img_size: Size of input images.
    - depth: Number of transformer layers.
    - num_heads: Number of attention heads in transformer layers.
    - mlp_ratio: Determines the hidden dimension size of the MLP layer based on the emb_size.
    - num_classes: Number of output classes for classification.
    """
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224, depth=12, num_heads=12, mlp_ratio=4, num_classes=2):
        super().__init__()
    
    # Patch embedding
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size)
        num_patches = (img_size // patch_size) ** 2
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, emb_size))
        
        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_size, nhead=num_heads, dropout=0.4),  
            num_layers=depth
        )        
        # Classifier head
        self.mlp_head = nn.Sequential(
            nn.Linear(emb_size, int(emb_size * mlp_ratio)),
            nn.BatchNorm1d(int(emb_size * mlp_ratio)),
            nn.Dropout(0.4),  
            nn.ReLU(),
            nn.Linear(int(emb_size * mlp_ratio), num_classes)
        )
        
    def forward(self, x):
        """
        Process the input image.
        
        Parameters:
        - x: The input image tensor of shape
        
        Returns:
        - Tensor of shape containing the class logits for each image in the batch.
        """
        x = self.patch_embed(x)
        
        # Adding class token and position embedding
        cls_tokens = self.cls_token.repeat(x.size(0), 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        x = x.mean(dim=1)
        
        # MLP head
        x = self.mlp_head(x)        
        return x
    

    

    
        
    