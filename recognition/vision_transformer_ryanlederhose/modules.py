import torch
import torch.nn as nn

class ImagePatcher(nn.Module):
    '''
    ImagePatcher

    This class defines the functions necessary to split the input image
    into a defined unmber of patches
    '''

    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, data):
        batch_size, channels, height, width = data.size()
        if (height % self.patch_size != 0) or (width % self.patch_size != 0):
            return 0

        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        num_patches = num_patches_h * num_patches_w

        patches = data.unfold(2, self.patch_size, self.patch_size). \
            unfold(3, self.patch_size, self.patch_size). \
            permute(0, 2, 3, 1, 4, 5). \
            contiguous(). \
            view(batch_size, num_patches, -1)
        
        return patches

class InputEmbedding(nn.Module):
    '''
    InputEmbedding
    
    This class defines the input embedding module of the ViT
    '''
    def __init__(self, args) -> None:
        super(InputEmbedding, self).__init__()
        self.batch_size = args.batch_size
        self.latent_size = args.latent_size
        self.n_channels = args.n_channels
        self.patch_size = args.patch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = self.n_channels * self.patch_size ** 2

        self.positionalEmbedding = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size)).to(self.device)
        self.classToken = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size)).to(self.device)
        self.linearProjection = nn.Linear(self.input_size, self.latent_size)

    def forward(self, input):
        input = input.to(self.device)

        # Get the image patcher object
        imagePatcher = ImagePatcher(patch_size=self.patch_size)

        # Project the patched images onto a linear plane using a FC linear layer
        linearProjection = self.linearProjection(imagePatcher(input)).to(self.device)

        # Define the class token
        self.classToken = nn.Parameter(torch.randn(linearProjection.shape[0], 1, self.latent_size)).to(self.device)

        # Concatenate the class token to the embedding tokens
        linearProjection = torch.cat((self.classToken, linearProjection), dim=1)

        # Add the positional embeddings to the input embeddings and class token
        linearProjection += self.positionalEmbedding[:linearProjection.shape[0], :linearProjection.shape[1] + 1, :]
        return linearProjection

class Encoder(nn.Module):
    '''
    Encoder
    
    This class defines the encoder block for the ViT
    '''
    def __init__(self, args) -> None:
        super(Encoder, self).__init__()

        self.dropout = args.dropout
        self.num_heads = args.num_heads
        self.latent_size = args.latent_size
        self.normLayer = nn.LayerNorm(self.latent_size)
        self.attention = nn.MultiheadAttention(self.latent_size, self.num_heads, dropout=self.dropout)
        self.encoderMLP = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size * 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_size * 4, self.latent_size),
            nn.Dropout(self.dropout)
        )
    
    def forward(self, embeddedPatches):
        # Normalise the embedded patches
        normalisation = self.normLayer(embeddedPatches)

        # Multi head attention output
        attentionOut = self.attention(normalisation, normalisation, normalisation)[0]

        # Second normalisation block
        normalisation = self.normLayer(attentionOut + embeddedPatches)

        # Encoder output
        return (self.encoderMLP(normalisation) + attentionOut + embeddedPatches)
    
class ViT(nn.Module):
    '''
    ViT
    
    This class defines the vision transformer architecture
    '''
    def __init__(self, args) -> None:
        super(ViT, self).__init__()

        self.dropout = args.dropout
        self.num_classes = args.num_classes
        self.num_encoders = args.num_encoders
        self.latent_size = args.latent_size
        
        self.encoders = nn.ModuleList([Encoder(args) for i in range(self.num_encoders)])
        self.embedding = InputEmbedding(args)
        self.MLP = nn.Sequential(
            nn.LayerNorm(self.latent_size),
            nn.Linear(self.latent_size, self.latent_size),
            nn.Linear(self.latent_size, self.num_classes)
        )

    def forward(self, input):
        # Get the embedding of the input
        encoderOut = self.embedding(input)

        # Loop through all the encoder blocks
        for layer in self.encoders:
            encoderOut = layer(encoderOut)

        # Extract the class token
        classTokenEmbedded = encoderOut[:, 0]

        # Output of MLP head is classification result
        return self.MLP(classTokenEmbedded)