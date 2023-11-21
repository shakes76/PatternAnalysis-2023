"""
Imports Here
"""
import numpy as np
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, heads, embed):
        super().__init__()
        self.heads = heads
        self.attn = nn.MultiheadAttention(embed, heads, batch_first=True)
        self.Q = nn.Linear(embed, embed, bias=False)
        self.K = nn.Linear(embed, embed, bias=False)
        self.V = nn.Linear(embed, embed, bias=False)

    def forward(self, x):
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        attnout, attnweights = self.attn(Q, K, V)
        return attnout

class TransBlock(nn.Module):
    def __init__(self, heads, embed, fflsize):
        super().__init__()
        self.fnorm = nn.LayerNorm(embed)
        self.snorm = nn.LayerNorm(embed)
        self.attn = Attention(heads, embed)
        self.ffl = nn.Sequential(
            nn.Linear(embed, fflsize),
            nn.GELU(),
            nn.Linear(fflsize, embed)
        )

    def forward(self, x):
        """
        Switching to pre-MHA LayerNorm is supposed to give better performance,
        this is used in other models such as LLMs like GPT. Gradients are meant
        to be stabilised. This is different to the original ViT paper.
        """
        x = x + self.attn(self.fnorm(x))
        x = x + self.ffl(self.snorm(x))
        return x
"""
Convolution pre
"""
class ConvLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        self.conv11 = nn.Conv3d(1, 48, kernel_size=(3,11,11), stride=(1,4,4), padding=(1,0,0))
        self.conv5 = nn.Conv3d(48, 192, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,0,0))

    def forward(self, imgs):
        x = self.conv11(imgs)
        x = self.relu(self.pool(x))
        x = self.conv5(x)
        x = self.relu(self.pool(x))
        return x
"""
Vision Transformer Class to create a vision transformer model
"""
class VisionTransformer(nn.Module):
    def __init__(self, classes=2, inputsize=(1,1,1), heads=2, embed=64, fflscale=2, nblocks=1):
        super().__init__()
        (self.N, self.Np, self.P) = inputsize
        """components"""
        self.proj = nn.Linear(self.P, embed)
        self.clstoken = nn.Parameter(torch.randn(1, 1, embed))
        self.posembed = self.embedding(self.Np+1, embed)
        self.transformer = nn.Sequential(
            *((TransBlock(heads, embed, int(fflscale*embed)),)*nblocks)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed),
            nn.Linear(embed, classes)
        )
        """convolutional components"""
        self.conv = ConvLayer()

    def embedding(self, npatches, embed, freq=10000): #10000 is described in ViT paper
        posembed = torch.zeros(npatches, embed)
        for i in range(npatches):
            for j in range(embed):
                if j % 2 == 0:
                    posembed[i][j] = np.sin(i/(freq**(j/embed)))
                else:
                    posembed[i][j] = np.cos(i/(freq**((j-1)/embed)))
        return posembed
    
    def forward(self, imgs): #assume size checking done by createPatches
        """Convolutional layer"""
        imgs = self.conv(imgs)
        imgs = imgs.flatten(2,4)
        """Linear Projection and Positional Embedding"""
        tokens = self.proj(imgs) #perform linear projection
        clstoken = self.clstoken.repeat(imgs.shape[0], 1, 1)
        tokens = torch.cat([clstoken, tokens], dim=1) #concat the class token
        x = tokens + self.posembed.repeat(imgs.shape[0], 1, 1) #add positional encoding
        """Transformer"""
        x = self.transformer(x)
        """Classification"""
        y = x[:,0]
        return self.classifier(y)