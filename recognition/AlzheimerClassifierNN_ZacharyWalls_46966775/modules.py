import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768):
        super().__init__()

        # 3D Convolution block for feature extraction
        self.conv3d_block = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, output_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # Global Average Pooling
            Rearrange("b c d h w -> b (c d h w)"),  # Flatten
        )

    def forward(self, x):
        return self.conv3d_block(x)


class TransformerBlock(nn.Module):
    def __init__(self, emb_size=768, drop_p=0.0, num_heads=8, forward_expansion=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

        self.attn = nn.MultiheadAttention(emb_size, num_heads)
        self.fc = nn.Sequential(
            nn.Linear(emb_size, forward_expansion * emb_size),
            nn.GELU(),
            nn.Linear(forward_expansion * emb_size, emb_size),
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
    def __init__(
        self,
        in_channels=3,
        patch_size=16,
        emb_size=768,
        img_size=240,
        num_blocks=12,
        num_heads=8,
        forward_expansion=4,
        num_classes=2,
    ):
        super().__init__()
        self.patch_emb = PatchEmbedding(in_channels, patch_size, emb_size)

        self.cls_token = nn.Parameter(randn(1, 1, emb_size))
        self.pos_emb = nn.Parameter(randn((img_size // patch_size) ** 2 + 1, emb_size))

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    emb_size, num_heads=num_heads, forward_expansion=forward_expansion
                )
                for _ in range(num_blocks)
            ]
        )

        self.norm = nn.LayerNorm(emb_size)
        self.fc = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = self.norm(x)

        cls_repr = x[:, 0]
        out = self.fc(cls_repr)

        return out
