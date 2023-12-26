"""
Module containing all the modules used in the ViT model.
This includes:
    PatchEmbed: Split image into patches and then embed them.
    Attention: Attention mechanism
    MLP: Multilayer perceptron
    Block: The Transformer Block!
    VisionTransformer: Pulls Everything Together - The Vision Transformer

Author: Felix Hall
Student number: 47028643
"""

import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """ 
    Split image into patches and then embed them.
    Parameters:
        image_size: size of image (it's a square)
        patch_size: size of patch (it's a square)
        in_channels: number of input channels (greyscale =1, rgb =3)
        embed_dim: dimension of embedding (how big an embedding of patch will be)
   
   Attributes:
   n_patches: number of patches in image
   or_j : nn.Conv2d: convolutional layer that splits image into patches
   """

    def __init__(self, image_size, patch_size, in_channels=1, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.n_patches = (image_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size)
        # take kernel size and stride, so when sliding kernel we never slide in overlapping way
        # so we never slide over same pixel twice, so kernel will fall into patches we dividing into

    def forward(self, x):
        """
        Run forward pass
        Parameters:
            x: input image tensor (B, C, H, W), B = num samples
        Output:
            tensor of flattened patches (B, n_patches, embed_dim)
        """
        x = self.proj(x) # (B, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2) # take the last 2D and flatten them into 1D
        x = x.transpose(1,2) #

        return x
    
class Attention(nn.Module):
    """
    Attention mechanism
    Parameters:
        dim: dimension of query and key
        heads: number of heads
        qkv_bias: if True, then we learn bias for q,k,v
        attn_p: dropout probability for attention layer
        proj_p: dropout probability for projection
    About dropout:
        Half of the neurons are dropped out randomly during training.
        This prevents neurons from co-adapting too much.
    Attributes:
        scale: scale factor for attention
        qkv: linear layer for q,k,v
        proj: linear layer for output projection
        attn_drop, proj_drop: dropout layers (nn.Dropout)
    """
    
    def __init__(self, dim, heads=8, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.dim = dim
        self.scale = self.heads ** -0.5 # not feeding small values into softmax - leading to exploding gradients

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p) # takes concatenated q,k,v and projects them to output dimension

    def forward(self, x):
        """
        Run forward pass
        Parameters:
            x: input tensor (B, n_patches + 1, embed_dim)
        Output:
            tensor after attention (B, n_patches + 1, embed_dim)
        """
        n_samples, n_tokens, dim = x.shape
        

        if dim != self.dim: # if input dimension is not equal to dimension of query and key
            raise ValueError
    
        qkv = self.qkv(x)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.heads, self.head_dim)
        # (B, n_patches + 1, 3, heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, B, heads, n_patches + 1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        k_t = k.transpose(-2, -1) # getting ready to compute dot
        dp = (q @ k_t) * self.scale # last two dims are compatible
        attn = dp.softmax(dim=-1) # over last dim, so as to create discrete prob dist that sums to 1 and be sued as weights in a weighted avg
        attn = self.attn_drop(attn)
        weighted_abg = attn @ v # computed weighted avg for all values
        weighted_abg = weighted_abg.transpose(1, 2) # swapped two dim
        weighted_abg = weighted_abg.flatten(2) # flattened last two dim

        x = self.proj(weighted_abg)
        x = self.proj_drop(x)

        return x, attn # returning attention weights for visualisation
    
class MLP(nn.Module):
    """
    Multilayer perceptron
    Parameters:
        in_features: number of input features
        hidden_features: number of hidden features
        out_features: number of output features
        p: dropout probability
    Attributes:
        fc: fully connected layer
        act: activation layer (nn.GELU) - Gaussian Error Linear Unit
        drop: dropout layer (nn.Dropout)
    
    """
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """
        Run forward pass
        Parameters:
            x: input tensor (B, n_patches + 1, embed_dim)
        Output:
            tensor after MLP (B, n_patches + 1, embed_dim)
        """

        x = self.fc1(x) # (B, n_patches + 1, hidden_features)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x
            
class Block(nn.Module):
    """
    Transformer Block!
    Parameters:
        dim: dimension of query and key
        heads: number of heads
        mlp_ratio: ratio of hidden dimension to embedding dimension
        qkv_bias: if True, then we learn bias for q,k,v
        p: dropout probability
    Attributes:
        norm1, norm2: layernorms
        attn: attention layer
        mlp: mlp layer
    """

    def __init__(self, dim,n_heads, mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6) # eps is a small value added to the denominator for numerical stability
        # layerNorm note: 
        self.attn = Attention(dim=dim, heads = n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio) # calculating abs value of hidden features
        self.mlp = MLP(in_features=dim, hidden_features=hidden_features, out_features=dim, p=p)

    def forward(self, x):
        """
        Run forward pass
        Parameters:
            x: input tensor (B, n_patches + 1, embed_dim)
        Output:
            tensor after transformer block (B, n_patches + 1, embed_dim)
        """
        x_res = x # residual connection
        x, attn = self.attn(self.norm1(x)) # get attn weights
        x = x + x_res # add residual back
        x = x + self.mlp(self.norm2(x)) 

        return x, attn # return attn weights for visualisation

class VisionTransformer(nn.Module):
    """
    Vision Transformer
    Parameters:
        img_size: size of image (it's a square)
        patch_size: size of patch (it's a square)
        in_channels: number of input channels (greyscale =1, rgb =3)
        n_classes: number of classes
        embed_dim: dimension of embedding (how big an embedding of patch will be)
        depth: number of transformer blocks
        n_heads: number of heads
        mlp_ratio: ratio of hidden dimension to embedding dimension
        qkv_bias: if True, then we learn bias for q,k,v
        p, attn_p: dropout probability
    Attributes:
        patch_embed: patch embedding layer
        cls_token: class token
        pos_emb: positional embedding layer which relates to where that patch is in the image
        pos_drop: dropout layer (nn.Dropout)
        blocks: transformer blocks (nn.ModuleList)
        norm: layernorm
        head: linear layer
    """

    def __init__(self, img_size=256, patch_size=16, in_channels=1, n_classes=1, embed_dim=768, depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # 1st 1 is batch size, 2nd 1 is number of patches; both are for convenience. 3rd is embedding dim
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)) # 1st 1 is batch size, 2nd 1 is number of patches, 3rd is embedding dim
        # above goal is to determine where in the image the patch is coming from. Also want to learn positional encoding for class token - hence +1
        self.pos_drop = nn.Dropout(p=p)

        # iteratively create transformer blocks, each block has a layer norm, attention layer, and mlp layer
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        """
        Run forward pass
        Parameters:
            x: input tensor (B, C, H, W), B = num samples
        Output:
            logits (B, n_classes)
        """
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(n_samples, -1, -1) # expand to match batch size - replicate over sample dim
        x = torch.cat((cls_token, x), dim=1) # concatenate along patch dim
        
        x = x + self.pos_embed # add positional embedding
        x = self.pos_drop(x)

        attns_weights = []

        for block in self.blocks:
            x, attn = block(x) # define forward pass for block in Block class
            attns_weights.append(attn)

        x = self.norm(x)

        cls_token_final = x[:, 0] # just taking the cls token
        x = self.head(cls_token_final)

        # we hope that embedding encodes enough info about the image that we can just use the cls token to classify the image
        return x, attns_weights # attn_weights returned for visualisation



