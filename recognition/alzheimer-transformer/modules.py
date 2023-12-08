'''
contains the source code of the components of the model. Each component is
implementated as a class or a function
'''

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Takes a 2D image and turns it into an input for the tranformer by flattening it
    into a 1D sequence learnable embedding vector as described in (Dosovitskiy et al., 2020)

    Args:
        in_channels (int): Number of colour channels. In the case of the ADNC, there are 3 (RGB)
            -> default = 3

        patch_size (int): Size of patches (pixels) that the image will be converted into
            -> default = 16

        embedding_dim (int): Size of embedding vector for each patch
            -> default = 768
    """

    def __init__(self,
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768):
        super().__init__()

        # Layer which converts an image into patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        # Flattens the patch feature maps to 1D
        # Only flatten the feature map dimension
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        return x_flattened.permute(0,2,1) # such that the embedding is on the final dimension



class ViT(nn.Module):
    """
    The main class for the Vision Transformer model
    """
    def __init__(self,
                 model_type:str='base',
                 num_classes:int=2,
                 img_size:int=224,
                 num_channels:int=3,
                 patch_size:int=16,
                 dropout:float=0.1):

        super().__init__()

        #3 types of model sizes according to (Dosovitskiy et al., 2020)
        if model_type == 'base':
            n_layers = 12
            hidden_D = 768
            mlp_size = 3072
            n_heads = 12

        elif model_type == 'large':
            n_layers = 24
            hidden_D = 1024
            mlp_size = 4096
            n_heads = 16

        elif model_type == 'huge':
            n_layers = 32
            hidden_D = 1280
            mlp_size = 5120
            n_heads = 16

        else:
            raise Exception("Model type must be one of: base, large, huge")

        # make sure img_size is divisble by patch_size
        assert img_size % patch_size == 0, f'Image size ({img_size}) is not divisble by patch_size ({patch_size})'

        #-----------------#
        # PATCH EMBEDDING #
        #-----------------#
        self.patch_embedding = PatchEmbedding(in_channels=num_channels,
                                         patch_size=patch_size,
                                         embedding_dim=hidden_D)
        #-------------#
        # CLASS TOKEN #
        #-------------#
        self.class_token = nn.Parameter(torch.randn(1, 1, hidden_D), requires_grad=True)

        #----------------------#
        # POSITIONAL EMBEDDING #
        #----------------------#
        num_patches = (img_size//patch_size)**2
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches+1, hidden_D))

        #------------------------------------#
        # DROPOUT AFTER POSITIONAL EMBEDDING #
        #------------------------------------#
        # according to appendix B.1 in (Dosovitskiy et al., 2020), dropout is used directly after adding positional to patch embeddings
        self.embedding_dropout = nn.Dropout(p=dropout)

        #-----------------------------#
        #  TRANSFORMER ENCODER STACK
        #-----------------------------#
        # this is to resemble the transformer enoder in figure 1 of (Dosovitskiy et al., 2020)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_D,
                nhead=n_heads,
                dim_feedforward=mlp_size,
                dropout=dropout,
                activation='gelu', # as per (Dosovitskiy et al., 2020)
                batch_first=True, # since the batch dimension is first in patch_embedding
                norm_first=True), # according to the tranformer encoder diagram in figure 1 of (Dosovitskiy et al., 2020),
            num_layers=n_layers)

        #----------#
        # MLP HEAD #
        #----------#

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=hidden_D), # from eq.4 of (Dosovitskiy et al., 2020)
            nn.Linear(in_features=hidden_D,
                      out_features=num_classes)
        )


    def forward(self, x):
        batch_size = x.shape[0]

        # make patch embedding
        x = self.patch_embedding(x)

        # stretch class token across whole batch size
        class_token = self.class_token.expand(batch_size, -1, -1)

        # concat class token to make it part of patch embedding (see X_class in eq.1 of Dosovitskiy et al., 2020)
        x = torch.cat((class_token, x), dim=1)

        # add positional embedding to x
        x = self.positional_embedding + x

        # use embedding_dropout (dropout after positional embedding)
        x = self.embedding_dropout(x)

        # input into trasnformer encoder stack
        x = self.transformer_encoder(x)

        # input only the 0th index through to the head
        x = self.mlp_head(x[:, 0])

        return x