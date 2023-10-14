import torch.nn as nn
from torch import randn, cat

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=1536):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)

class TransformerBlock(nn.Module):
    def __init__(self, emb_size=1536, drop_p=0.1, num_heads=8, forward_expansion=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        
        self.attn = nn.MultiheadAttention(emb_size, num_heads)
        
        self.fc = nn.Sequential(
            nn.Linear(emb_size, forward_expansion * emb_size),
            nn.GELU(),
            nn.Linear(forward_expansion * emb_size, forward_expansion * emb_size),
            nn.GELU(),
            nn.Linear(forward_expansion * emb_size, emb_size)
        )
        
        self.drop = nn.Dropout(drop_p)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + self.drop(attn_out)
        x = self.norm1(x)
        
        fc_out = self.fc(x)
        x = x + self.drop(fc_out)
        x = self.norm2(x)
        
        return x

class ViT(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=1536, img_size=240, num_blocks=12, num_heads=8, forward_expansion=4, num_classes=2, drop_p=0.1):
        super().__init__()
        self.patch_emb = PatchEmbedding(in_channels, patch_size, emb_size)
        self.emb_drop = nn.Dropout(drop_p) 

        self.cls_token = nn.Parameter(randn(1, 1, emb_size))
        self.pos_emb = nn.Parameter(randn((img_size // patch_size) ** 2 + 1, emb_size))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(emb_size, num_heads=num_heads, forward_expansion=forward_expansion)
            for _ in range(num_blocks)
        ])
        
        self.norm = nn.LayerNorm(emb_size)
        self.fc1 = nn.Linear(emb_size, emb_size)
        self.fc2 = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        B, _, _, _ = x.shape
        x = self.patch_emb(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        x = cat([cls_tokens, x], dim=1)
        x += self.pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        
        cls_repr = x[:, 0]
        out = self.fc1(cls_repr)
        out = self.fc2(out)
        
        return out

