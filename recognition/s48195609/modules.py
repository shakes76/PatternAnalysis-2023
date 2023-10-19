"""
modules.py

Components of the visual transformer model.

Author: Atharva Gupta
Date Created: 17-10-2023
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchLayer(nn.Module):
    """
    Layer for shifting inputted images and transforming images into patches.
    """
    def __init__(self, image_size, patch_size, num_patches, projection_dim):
        super(PatchLayer, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.half_patch = patch_size // 2
        self.flatten_patches = nn.Flatten(1)
        self.projection = nn.Linear(self.num_patches * self.projection_dim, self.num_patches * self.projection_dim)
        self.layer_norm = nn.LayerNorm(self.num_patches * self.projection_dim)

   
    def forward(self, images):
        images = torch.cat([
            images,
            self.shift_images(images, mode='left-up'),
            self.shift_images(images, mode='left-down'),
            self.shift_images(images, mode='right-up'),
            self.shift_images(images, mode='right-down')
        ], dim=1)

        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(patches.size(0), patches.size(1), -1)
        patches = patches.permute(0, 2, 1)

        flat_patches = self.flatten_patches(patches)
        tokens = self.layer_norm(flat_patches)
        tokens = self.projection(tokens)

        return (tokens, patches)

class EmbedPatch(nn.Module):
    """
    Layer for projecting patches into a vector. Also adds a learnable position embedding to the projected vector.
    """
    def __init__(self, num_patches, projection_dim):
        super(EmbedPatch, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.projection_dim))

    def forward(self, patches):
        positions = torch.arange(self.num_patches, device=patches.device).unsqueeze(0)
        position_embedding = self.position_embedding
        return patches + position_embedding

class MultiHeadAttentionLSA(nn.MultiheadAttention):
    """
    Multi Head Attention layer for the transformer encoder block, but with the addition of using Local Self Attention to improve finer-level feature learning.
    """
    def __init__(self, embed_dim, num_heads, **kwargs):
        super(MultiHeadAttentionLSA, self).__init__(embed_dim, num_heads, **kwargs)
        self.tau = nn.Parameter(math.sqrt(float(embed_dim), requires_grad=True))

    def forward(self, query, key, value, attn_mask=None, bias_k=None, bias_v=None):
        query = query / self.tau
        return super(MultiHeadAttentionLSA, self).forward(query, key, value, attn_mask, bias_k, bias_v)
