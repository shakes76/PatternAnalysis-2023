"""
"ViT" class is a wrapper class for the vit_b_16 model and my custom VisionTransformer module.

"VisionTransformer" module and associated modules created by looking at code from 
vision_transformer.py file in the torchvision module of PyTorch.

URL: https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
"""

import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights
from collections import OrderedDict


class ViT(nn.Module):
    """
    ViT wrapper class. Includes vit_b_16 architecture with MLP head to suit the task (i.e. 2 output features, AD or NC)
    """
    def __init__(self):
        super().__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.model.heads.head = nn.Linear(in_features=768, out_features=2)

    def forward(self, x):
        return self.model.forward(x)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, patch_size: int):
        super().__init__()
        self.conv_proj = nn.Conv2d(
                in_channels=in_channels, 
                out_channels=hidden_dim, 
                kernel_size=patch_size, 
                stride=patch_size
        )

    def forward(self, x):
        x = self.conv_proj(x)
        # flatten h and w into h*w
        x = x.flatten(2)
        # permute so that embedding/hidden dimension is last
        x = x.permute(0, 2, 1)
        return x
    

class MLP(nn.Module):
    '''
    Multi-layer perceptron block, inspired by the video "Vision Transformer in PyTorch"
    URL: https://www.youtube.com/watch?v=ovB0ddFtzzA
    '''
    def __init__(
            self, 
            mlp_dim: int,
            hidden_dim: int,
            dropout: int
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    '''
    An encoder block for the vision transformer encoder. 
    '''
    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: int,
            attention_dropout: int
        ):
        super().__init__()
        # Layer norm
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)

        # Multi head attention 
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(mlp_dim, hidden_dim, dropout)
    
    def forward(self, input):
        x = self.layer_norm_1(input)
        x, _ = self.attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input
        
        y = self.layer_norm_2(x)
        y = self.mlp(y)

        return x + y

class Encoder(nn.Module):
    '''
    Vision Transformer model encoder.
    '''
    def __init__(
            self, 
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int, 
            dropout: int,
            attention_dropout: int,
            ):
        super().__init__()
        # For look of EncoderBlocks
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout
            )
        self.layers = nn.Sequential(layers)
        self.norm_layer = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.layers(x)
        x = self.norm_layer(x)
        return x

class VisionTransformer(nn.Module):
    '''
    Vision transformer as per https://arxiv.org/abs/2010.11929
    '''
    def __init__(
            self, 
            image_size: int = 224, 
            patch_size: int = 16, 
            num_layers: int = 12,
            num_heads: int = 12,
            hidden_dim: int = 768,
            mlp_dim: int = 3072,
            num_classes: int = 2,
            in_channels: int = 1,
            attention_dropout: float = 0.,
            dropout: float = 0.
    ):
        super().__init__()
        # Patch embedding (conv_proj)
        self.patch_embedding = PatchEmbedding(in_channels, hidden_dim, patch_size)
        self.n_patches = (image_size // patch_size) ** 2
        # Class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.n_patches += 1
        # Positional embedding
        self.positional_embedding = nn.Parameter(torch.empty(1, self.n_patches, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        # Encoder
        self.transformer_encoder = Encoder(num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout)  
        # Classifier head
        self.classifier_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        # Patch embedding
        x = self.patch_embedding(x)
        # Class token
        batch_class_token = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        # Position embedding
        x = self.positional_embedding + x
        x = self.dropout(x)
        # Transformer encoder
        x = self.transformer_encoder(x)
        # Classifier head
        x = self.classifier_head(x[:, 0])
        return x