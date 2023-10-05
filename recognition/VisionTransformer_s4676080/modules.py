# this is the modules file

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  
        return x
    
class VisionTransformer(nn.Module):
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
            nn.Dropout(0.4),  
            nn.ReLU(),
            nn.Linear(int(emb_size * mlp_ratio), num_classes)
        )
    