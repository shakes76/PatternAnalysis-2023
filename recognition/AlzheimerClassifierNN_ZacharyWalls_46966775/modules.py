import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# Helper function to handle tuples
def pair(t):
    """
    Helper function to handle tuples.

    Parameters:
        t (int or tuple): Input which can be a single integer or a tuple.

    Returns:
        tuple: A tuple containing two values.
    """
    return t if isinstance(t, tuple) else (t, t)


# Feed Forward Neural Network
class FeedForward(nn.Module):
    """
    Feed Forward Neural Network.

    Parameters:
        dim (int): Dimension of the input.
        hidden_dim (int): Dimension of the hidden layer.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
    """

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Forward pass of the Feed Forward Network.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.net(x)


# Multi-head Attention Mechanism
class Attention(nn.Module):
    """
    Multi-head Attention Mechanism.

    Parameters:
        dim (int): Dimension of the input.
        heads (int, optional): Number of attention heads. Defaults to 8.
        dim_head (int, optional): Dimension of each attention head. Defaults to 64.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        """
        Forward pass of the Attention mechanism.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# Transformer (comprising multiple layers of multi-head attention and feed-forward networks)
class Transformer(nn.Module):
    """
    Transformer comprising multiple layers of multi-head attention and feed-forward networks.

    Parameters:
        dim (int): Dimension of the input.
        depth (int): Number of layers in the transformer.
        heads (int): Number of attention heads.
        dim_head (int): Dimension of each attention head.
        mlp_dim (int): Dimension of the hidden layer in the feed forward network.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        """
        Forward pass of the Transformer.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    """
    Vision Transformer (ViT) model.

    Parameters:
        image_size (int or tuple): Size of the input image.
        image_patch_size (int or tuple): Size of each image patch.
        frames (int): Number of frames.
        frame_patch_size (int): Size of each frame patch.
        num_classes (int): Number of target classes.
        dim (int): Dimension of the input.
        depth (int): Number of layers in the transformer.
        heads (int): Number of attention heads.
        mlp_dim (int): Dimension of the hidden layer in the feed forward network.
        pool (str, optional): Pooling type, either "cls" for cls token or "mean" for mean pooling. Defaults to "cls".
        channels (int, optional): Number of channels in the input image. Defaults to 3.
        dim_head (int, optional): Dimension of each attention head. Defaults to 64.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        emb_dropout (float, optional): Dropout rate for embeddings. Defaults to 0.0.
    """

    def __init__(
        self,
        *,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."
        assert (
            frames % frame_patch_size == 0
        ), "Frames must be divisible by frame patch size"

        num_patches = (
            (image_height // patch_height)
            * (image_width // patch_width)
            * (frames // frame_patch_size)
        )
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)",
                p1=patch_height,
                p2=patch_width,
                pf=frame_patch_size,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

        # Initialize linear layers and LayerNorm in your model
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(
                    module.weight
                )  # Use Xavier initialization for linear layers
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  # Initialize biases to zeros

            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)  # Initialize LayerNorm weights to 1
                nn.init.constant_(
                    module.bias, 0
                )  # Initialize LayerNorm biases to zeros

    def forward(self, image):
        x = self.to_patch_embedding(image)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
