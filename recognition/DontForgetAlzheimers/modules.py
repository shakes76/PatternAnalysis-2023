"""
ViT model
"""
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from dataset import AlzheimerDataset
from dataset import transform

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=16, embed_size=768):
        """Embeds image patches into vectors."""
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_size, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(start_dim=2)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))

    def forward(self, x):
        x = self.projection(x) 
        x = x.transpose(1, 2) 
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        return torch.cat((cls_tokens, x), dim=1)



class TransformerEncoder(nn.Module):
    """Single layer of the Transformer encoder."""
    def __init__(self, embed_size=768, num_heads=8, feedforward_dim=2048, drop_p=0.1):
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_size, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_size)
        )
        self.drop = nn.Dropout(drop_p)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + self.drop(attn_output)
        x = self.norm1(x)

        ff_output = self.feedforward(x)
        x = x + self.drop(ff_output)
        x = self.norm2(x)
        return x


class ViT(nn.Module):
    """Model for image classification."""
    def __init__(self, in_channels=1, patch_size=16, embed_size=768, img_size=224, num_layers=12, num_heads=8, feedforward_dim=2048, num_classes=2):
        super(ViT, self).__init__()
        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (img_size // patch_size) ** 2
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_size)
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_size))
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoder(embed_size, num_heads, feedforward_dim)
            for _ in range(num_layers)
        ])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x + self.positional_embedding
        for encoder in self.transformer_encoders:
            x = encoder(x)
        return self.mlp_head(x[:, 0])