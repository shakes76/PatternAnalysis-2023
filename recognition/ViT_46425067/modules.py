"""

"""
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    """Split images into patches
    """
    def __init__(self, img_size: int, patch_size:int, embed_dim=768, in_channels=1, linear_mode=True):
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
                nn.LayerNorm(self.patch_size * self.patch_size * in_channels),
                nn.Linear(self.patch_size * self.patch_size * in_channels, embed_dim),
                nn.LayerNorm(embed_dim)
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
    def __init__(self, dim, num_heads=12, qkv_bias=True, drop_prob=0.1):
        """initialise the self-attention mechanism for the transformEncoder

        Args:
            dim (int):              input, output dimension per feature
            num_heads (int):        number of attention heads
            qkv_bias (bool):        if we include bias in qkv projections
            drop_prob (float):      dropout probability
        params
            scale (float):          normalising constant for dot product
            qkv (nn.Linear):        linear projection for query, key, value
            proj (nn.Linear):       linear mapping of concatenated attention head output to 
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        # Makes the multi-head attention output dim the same as input dim
        self.head_dim = dim // num_heads
        # prevents small gradients 
        self.scale = self.head_dim ** -0.5
        # could do the query, key, value separately
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.attend = nn.Softmax(dim= -1)
        #dropout for attention and project layers
        self.attn_drop = nn.Dropout(drop_prob)
        self.proj_drop = nn.Dropout(drop_prob)
        
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
        if dim != self.dim:
            raise ValueError
        
        #generate queries, keys, values
        qkv = self.qkv(x) # (num_batches, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(num_batches, num_patches, 3 , self.num_heads, self.head_dim)
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
        super().__init__()
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
    

class TransformEncoder(nn.Module):
    """transformer encoder block
    """
    def __init__(self, dim:int, num_heads:int, mlp_ratio=4.0, qkv_bias=True, drop_prob=0.1):
        """intialise the transformer encoder block

        Args:
            dim (int):          the input dimension
            num_heads (int):    number of attention heads
            mlp_ratio (float):  ratio of hidden units to dim of the feed forward layer
            qkv_bias (bool):    if we want a bias term in q,k,v projections
            drop_prob (float):  drop out probability
        """
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attention_block = Attention(dim=dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              drop_prob=drop_prob)
        
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        
        #initialise feed forward layer
        hidden_units = int(dim * mlp_ratio)
        self.feed_forward = FeedForward(in_out_features=dim,
                                        hidden_units=hidden_units)
        
    def forward(self, x):
        """forward pass

        Args:
            x (torch.Tensor): input shape (n_batch, n_patch + 1, dim)
        
        Returns:
            (torch.Tensor): output shape (n_batch, n_patch + 1, dim)
        """
        x = self.attention_block(self.layer_norm1(x)) + x
        x = self.feed_forward(self.layer_norm2(x)) + x
        
        return x
    
    
class ViT(nn.Module):
    def __init__(self, img_size:int,
                    patch_size:int,
                    img_channels:int,
                    num_classes:int,
                    embed_dim:int,
                    depth:int,
                    num_heads:int,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    drop_prob=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size=img_size,
                                            patch_size=patch_size,
                                            embed_dim=embed_dim,
                                            in_channels=img_channels,
                                            linear_mode=False) #TODO: try changing to linear mode
        #class token to determine which class the image belongs to
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) #zeros or randn
        #positional information of patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_prob)
        
        #transform encoder blocks
        self.encoders = nn.ModuleList([]) #TODO: move this into the transform encoder class
        for _ in range(depth):
            self.encoders.append(TransformEncoder(dim=embed_dim,
                                                    num_heads=num_heads,
                                                    mlp_ratio=mlp_ratio,
                                                    qkv_bias=qkv_bias,
                                                    drop_prob=drop_prob))
        self.norm_layer = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        num_batch = x.shape[0]
        # convert images into patches
        x = self.patch_embed(x)
        # concate the class token
        class_token = self.class_token.expand(num_batch, -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        #pass x through encoders
        for encoder in self.encoders:
            x = encoder(x)
        x = self.norm_layer(x)
        
        #get only the class token for output
        output = x[:, 0] 
        x = self.head(output)
        return x
    
    
class ViT_torch(nn.Module):
    def __init__(self, img_size:int,
                    patch_size:int,
                    img_channels:int,
                    num_classes:int,
                    embed_dim:int,
                    depth:int,
                    num_heads:int,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    drop_prob=0.1,
                    linear_embed=False):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size=img_size,
                                            patch_size=patch_size,
                                            embed_dim=embed_dim,
                                            in_channels=img_channels,
                                            linear_mode=linear_embed) #TODO: try changing to linear mode
        #class token to determine which class the image belongs to
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim)) #zeros or randn
        #positional information of patches
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_prob)
        
        #transform encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                    nhead=num_heads,
                                                    dim_feedforward=embed_dim*mlp_ratio,
                                                    activation='gelu',
                                                    batch_first=True,
                                                    norm_first=True, 
                                                    layer_norm_eps=1e-6)
        
        self.transform_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                                        num_layers=depth)
        
        self.norm_layer = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        num_batch = x.shape[0]
        # convert images into patches
        x = self.patch_embed(x)
        # concate the class token
        class_token = self.class_token.expand(num_batch, -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        #pass x through encoders
        x = self.transform_encoder(x)
        
        #get only the class token for output
        output = x[:, 0]
        x = self.head(self.norm_layer(output))
        return x