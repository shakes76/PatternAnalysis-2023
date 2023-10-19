import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Embedder(nn.Module):
    def __init__(self, patch_size, dim):
        super(Embedder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(dim)
        )

    def forward(self, inputs):
        encoded = self.encoder(inputs)
        flattened = encoded.view(encoded.size(0), encoded.size(1), -1)
        return flattened

class VisionModel(nn.Module):
    def __init__(self, num_classes, image_size, patch_size, num_patches, dim, depth, num_heads, mlp_dim):
        super(VisionModel, self).__init__()

        self.embedder = Embedder(patch_size, dim)
        self.token = nn.Parameter(torch.randn(1, 1, dim))
        self.positions = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.transformer = nn.Transformer(
            d_model=dim,
            nhead=num_heads,
            num_encoder_layers=depth,
            dim_feedforward=mlp_dim,
            dropout=0.1
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, inputs):
        embedded = self.embedder(inputs)

        token = self.token.repeat(embedded.size(0), 1, 1)
        embedded = torch.cat([token, embedded], dim=1)
        embedded += self.positions

        encoded = self.transformer(embedded)
        aggregated = torch.mean(encoded, dim=1)

        return self.mlp_head(aggregated)



