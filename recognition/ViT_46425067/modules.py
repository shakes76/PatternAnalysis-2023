import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    """Split images into patches
    """
    def __init__(self, img_size: int, patch_size:int, embed_dim=768, in_channels=1, linear_mode=False):
        """
        initialise patch embedding layer for the ViT

        Args:
            img_size (int): H or W of image (should be a square number)
            patch_size (int): H or W of patch size (should be a square number)
            embed_dim (int, optional): Size of patch embedding stays constant across entire network. Defaults to 768.
            in_channels (int, optional): RGB channel of image. Defaults to 1 for greyscale.
            linear_mode (bool): whether to use linear projection method or convolution
        """
        super().__init__()
        #TODO: check img_size is a squre number
        #TODO: check patch_size is a sqaure number
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.linear_mode = linear_mode
        #kernel is the same size of the patch size and will never overlap since
        self.projection = nn.Conv2d(in_channels=in_channels, 
                                    out_channels=embed_dim,
                                    kernel_size=self.patch_size,
                                    stride=self.patch_size)
        
        # embedding using linear layers
        if self.linear_mode:
            self.projection = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                            p1=self.patch_size, p2=self.patch_size),
                # TODO: add layerNorm?
                nn.Linear(self.patch_size * self.patch_size * in_channels, embed_dim)
                # TODO: add LayerNorm?
            )
        
    def forward(self, x):
        """forward pass

        Args:
            x (tensor): image tensor of (batch_size, img_channels, H, W)
        
        Returns:
            tensor: (batch_size, num_patches, embed_dim)
        """
        x = self.projection(x)
        
        #if not using the linear project we need to reshape the tensor after convolution
        if not self.linear_mode:
            x = x.flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop_prob=0.1, proj_drop_prob=0.1):
        """initialise the self-attention mechanism for the transformEncoder

        Args:
            dim (int):              input, output dimension per feature
            num_heads (int):        number of attention heads
            qkv_bias (bool):        if we include bias in qkv projections
            attn_drop_prob (float): dropout probability for qkv 
            proj_drop_prob (float): dropout probability for output 
        params
            scale (float):          normalising constant for dot product
            qkv (nn.Linear):        linear projection for query, key, value
            proj (nn.Linear):       linear mapping of concatenated attention head output to 
        """
        self.dim = dim
        self.num_heads = num_heads
        # Makes the multi-head attention output dim the same as input dim
        self.head_dim = dim // num_heads
        # prevents small gradients 
        self.scale = self.head_dim ** -0.5
        # could do the query, key, value separately
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        #dropout for attention and project layers
        self.attn_drop = nn.Dropout(attn_drop_prob)
        self.proj_drop = nn.Dropout(proj_drop_prob)
        
        # maps the multi-head attention output to a new space
        self.proj = nn.Linear(dim, dim) 
        
    def forward(self, x):
        """forward pass

        Args:
            x (torch.Tensor): input tensor with shape (num_batches, num_patches + 1, dim) #TODO: CHECK THAT THIS DIM IS embeded dim?
                the + 1 indicates the cls token added to the patch embeddings
        Return:
            (torch.Tensor): tensor of shape (num_batches, num_patches + 1, dim)
        """
        num_batches, num_patches, dim = x.shape
        print(x.shape)
        if dim != self.dim:
            raise ValueError
        
        #generate queries, keys, values
        qkv = self.qkv(x) # (num_batches, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(num_batches, num_patches, 3 ,self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) #(3, n_batch, n_heads, n_patch, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        dot_prod = torch.matmul(q, k.transpose(-2, -1)) * self.scale #(batch, heads, patch, patch)
        
        #calculate probabilities
        attn = self.attend(dot_prod)
        attn = self.attn_drop(attn)
        
        #calculate weighted average
        weighted_avg = torch.matmul(attn, v).transpose(1, 2) # (batch, patch, heads, head_dim)
        weighted_avg = weighted_avg.flatten(2) # (batch, patches, dim)
        
        # linearly project onto new space
        x = self.proj(weighted_avg)
        x = self.proj_drop(x)
        
        return x


class FeedForward(nn.Module):
    """
    MLP layer in the transform encoder block
    """
    def __init__(self, in_out_features:int, hidden_units:int, drop_prob=0.1):
        """intialise the MLP layer

        Args:
            in_out_features (int): dimension of the input and output
            hidden_units (int): number of hiddent units used in linear layer        
            drop_prob (float, optional): dropout probabilitiy. Defaults to 0.1.
        """
        super().__init()
        self.net = nn.Sequential(
            nn.LayerNorm(in_out_features),
            nn.Linear(in_out_features, hidden_units),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_units, in_out_features),
            nn.Dropout(drop_prob),
        )
        
    def forward(self, x):
        return self.net(x)