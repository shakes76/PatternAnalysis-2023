"""
modules.py

Contains the perceiver model.

Author: Atharva Gupta
Date Created: 17-10-2023
"""

import torch
import torch.nn as nn

class CustomCrossAttention(nn.Module):
    def __init__(self, d_latents):
        super(CustomCrossAttention, self).__init__()
        self.attn = nn.MultiheadAttention(d_latents, 4, batch_first=True)
        self.o_l = nn.Linear(d_latents, d_latents)

    def forward(self, latent, kv):
        # Perform cross-attention
        attn_output = self.attn(latent, kv, kv)[0]
        output = self.o_l(attn_output)
        return output

class CustomSelfAttention(nn.Module):
    def __init__(self, d_latents):
        super(CustomSelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(d_latents, 4, batch_first=True)
        self.o_l = nn.Linear(d_latents, d_latents)

    def forward(self, latent):
        # Perform self-attention
        attn_output = self.attn(latent, latent, latent)[0]
        output = self.o_l(attn_output)
        return output

class CustomLatentTransformer(nn.Module):
    def __init__(self, d_latents, depth):
        super(CustomLatentTransformer, self).__init__()
        self.ff = nn.ModuleList([CustomMLP(d_latents) for _ in range(depth)])
        self.sa = nn.ModuleList([CustomSelfAttention(d_latents) for _ in range(depth)])
        self.depth = depth
        self.ln1 = nn.LayerNorm(d_latents)
        self.ln2 = nn.LayerNorm(d_latents)

    def forward(self, x):
        latent = x
        for i in range(self.depth):
            # Perform self-attention and feed-forward
            latent = self.sa[i](self.ln1(latent)) + latent
            latent = self.ff[i](self.ln2(latent)) + latent
        return latent

class CustomMLP(nn.Module):
    def __init__(self, d_latents):
        super(CustomMLP, self).__init__()
        self.ln = nn.LayerNorm(d_latents)
        self.l1 = nn.Linear(d_latents, d_latents)
        self.l2 = nn.Linear(d_latents, d_latents)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.ln(x)
        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        return x

class CustomBlock(nn.Module):
    def __init__(self, d_latents):
        super(CustomBlock, self).__init__()
        self.ca = CustomCrossAttention(d_latents)
        self.ff = CustomMLP(d_latents)
        self.ln1 = nn.LayerNorm(d_latents)
        self.ln2 = nn.LayerNorm(d_latents)

    def forward(self, x, data):
        attn = self.ca(self.ln1(x), data)
        x = attn + x
        x = self.ff(self.ln2(x) + x)
        return x

class CustomOutput(nn.Module):
    def __init__(self, n_latents, n_classes=2):
        super(CustomOutput, self).__init__()
        self.project = nn.Linear(n_latents, n_classes)

    def forward(self, x):
        # Calculate the average across dimensions
        average = torch.mean(x, dim=2)
        logits = self.project(average)
        return logits

class CustomPerceiver(nn.Module):
    def __init__(self, n_latents, d_latents, transformer_depth, n_cross_attends):
        super(CustomPerceiver, self).__init__()
        self.depth = n_cross_attends

        unique_latent = torch.empty(n_latents, d_latents)
        nn.init.trunc_normal_(unique_latent, std=0.02)
        self.unique_latent = nn.Parameter(unique_latent)

        self.custom_cross_attends = nn.ModuleList([CustomBlock(d_latents) for _ in range(self.depth)])
        self.custom_latent_transformers = nn.ModuleList([CustomLatentTransformer(d_latents, transformer_depth) for _ in range(self.depth)])
        self.custom_output = CustomOutput(n_latents)

        self.custom_image_project = nn.Linear(1, d_latents)

        self.custom_pe = nn.Parameter(torch.empty(1, 240*240, d_latents))
        nn.init.normal_(self.custom_pe)

    def forward(self, data):
        b, _, _, _ = data.size()
        flat_img = torch.flatten(data, start_dim=1)[:, :, None]
        custom_proj_img = self.custom_image_project(flat_img) + self.custom_pe.repeat(b, 1, 1)

        x = self.unique_latent.repeat(b, 1, 1)

        for i in range(self.depth):
            x = self.custom_cross_attends[i](x, custom_proj_img)
            x = self.custom_latent_transformers[i](x)
        custom_output = self.custom_output(x)
        return custom_output
