import torch
import torch.nn as nn

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
        self.input_size = args.hidden_size
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=args.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding='valid'
        )

        self.positionalEmbedding = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size)).to(self.device)
        self.classToken = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size)).to(self.device)
        self.linearProjection = nn.Linear(self.input_size, self.latent_size)

    def forward(self, input):
        input = input.to(self.device)

        # Patch the image using a convolutional layer
        patches = self.conv1(input)
        seq_len = (input.shape[2] // self.patch_size) * (input.shape[3] // self.patch_size)
        imagePatches = torch.reshape(patches, [-1, seq_len, self.input_size])

        # Project the patched images onto a linear plane using a FC linear layer
        linearProjection = self.linearProjection(imagePatches).to(self.device)

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
            nn.Linear(self.latent_size, self.latent_size),
            nn.GELU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.latent_size, self.latent_size),
            nn.GELU(),
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

        # Output of MLP head is classification result
        out = self.MLP(torch.mean(encoderOut, dim=1))
        return out