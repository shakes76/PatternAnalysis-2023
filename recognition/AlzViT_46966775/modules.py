import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# Model Architecture is largely based on the original paper
# A. Dosovitsky et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"

# Modules


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        """
        Initialize a residual addition block.

        Args:
            fn (nn.Module): The function to apply.
        """
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        """
        Forward pass of the residual addition block.

        Args:
            x (Tensor): Input tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: Output tensor.
        """
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        patch_size: int = 14,
        emb_size: int = 1536,
        img_size: int = 224,
    ):
        """
        Initialize the patch embedding layer.

        Args:
            in_channels (int): Number of input channels.
            patch_size (int): Patch size.
            emb_size (int): Embedding size.
            img_size (int): Image size.
        """
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # Using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size, dtype=torch.float32))
        self.positions = nn.Parameter(
            torch.randn(
                (img_size // patch_size) ** 2 + 1, emb_size, dtype=torch.float32
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the patch embedding layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, "() n e -> b n e", b=b)
        # Prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # Add position embedding
        x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 1536, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize the multi-head attention layer.

        Args:
            emb_size (int): Embedding size.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # Fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Forward pass of the multi-head attention layer.

        Args:
            x (Tensor): Input tensor.
            mask (Tensor, optional): Attention mask tensor.

        Returns:
            Tensor: Output tensor.
        """
        # Split keys, queries and values in num_heads
        qkv = rearrange(
            self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3
        )
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # Sum up over the last axis
        energy = torch.einsum(
            "bhqd, bhkd -> bhqk", queries, keys
        )  # Batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # Sum up over the third axis
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


# Modified compared to paper
class FeedForward(nn.Module):
    def __init__(self, dim, expansion=4, drop_p=0.1):
        """
        Initialize the feedforward layer.

        Args:
            dim (int): Input dimension.
            expansion (int): Expansion factor.
            drop_p (float): Dropout rate.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(drop_p),
        )

    def forward(self, x):
        """
        Forward pass of the feedforward layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self.net(x)


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        emb_size: int = 1536,
        drop_p: float = 0.1,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.05,
        **kwargs
    ):
        """
        Initialize a transformer encoder block.

        Args:
            emb_size (int): Embedding size.
            drop_p (float): Dropout rate.
            forward_expansion (int): Feedforward expansion factor.
            forward_drop_p (float): Feedforward dropout rate.
        """
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, **kwargs),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForward(
                        emb_size, expansion=forward_expansion, drop_p=forward_drop_p
                    ),
                    nn.Dropout(drop_p),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 10, **kwargs):
        """
        Initialize the transformer encoder.

        Args:
            depth (int): Number of transformer blocks.
            **kwargs: Additional keyword arguments for transformer blocks.
        """
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 1536, n_classes: int = 2):
        """
        Initialize the classification head.

        Args:
            emb_size (int): Embedding size.
            n_classes (int): Number of output classes.
        """
        super().__init__(
            Reduce("b n e -> b e", reduction="mean"),  # Mean reduction applied
            nn.Identity(),  # To Latent
            nn.Sequential(
                nn.LayerNorm(emb_size), nn.Linear(emb_size, n_classes)
            ),  # MLP Head
        )


class ViT(nn.Sequential):
    def __init__(
        self,
        in_channels: int = 1,
        patch_size: int = 14,
        emb_size: int = 1536,
        img_size: int = 224,
        depth: int = 10,
        n_classes: int = 2,
        **kwargs
    ):
        """
        Initialize the Vision Transformer model.

        Args:
            in_channels (int): Number of input channels.
            patch_size (int): Patch size.
            emb_size (int): Embedding size.
            img_size (int): Image size.
            depth (int): Number of transformer blocks.
            n_classes (int): Number of output classes.
        """
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth=depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes),
        )
