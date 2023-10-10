import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=10), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=7), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=4), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=4), nn.ReLU()
        )

        self.embedding = nn.Sequential(
            nn.Linear(128*24*24, 512),
            nn.Linear(512, 1),

            nn.ReLU()
        )

        # Classify whether the images belong to the same class or different classes
        self.fc = nn.Sequential(
            nn.Sigmoid()
        )

        # self.conv1 = nn.Conv2d(3, 64, 10)
        # self.conv2 = nn.Conv2d(64, 128, 7)
        # self.conv3 = nn.Conv2d(128, 256, 4)
        # self.conv4 = nn.Conv2d(256, 512, 4)

        # self.pool1 = nn.MaxPool2d(2, stride=2)
        # self.pool2 = nn.MaxPool2d(2, stride=2)
        # self.pool3 = nn.MaxPool2d(2, stride=2)
        # self.pool4 = nn.MaxPool2d(2, stride=2)

        # self.relu1 = nn.ReLU(inplace=True)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.relu3 = nn.ReLU(inplace=True)
        # self.relu4 = nn.ReLU(inplace=True)

        # self.fc1 = nn.Linear(128*24*24, 512)
        # self.fc2 = nn.Linear(512, 1)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.sub_forward(x1)
        h2 = self.sub_forward(x2)
        diff = torch.abs(h1 - h2)
        scores = self.fc(diff)
        
        return scores

    def sub_forward(self, x):
        x = self.conv(x)
        x = self.embedding(x)
        return x
    
class Residual(nn.Module):
    def __init__(self, in_channels, n_hidden, n_residual):
        super(Residual, self).__init__()

        self.residual = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=n_hidden, 
                out_channels=n_residual, 
                kernel_size=3, 
                stride=1, 
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_residual, 
                out_channels=n_hidden, 
                kernel_size=1, 
                stride=1, 
                padding=0
            ),
        )
    
    def forward(self, out):
        return out + self.residual(out)


class Encoder(nn.Module):
    """
    Encoder module for the VQ-VAE model.

    The encoder consists of 2 strided convolutional layers with stride 2 and 
    window size 4 × 4, followed by two residual 3 × 3 blocks (implemented as 
    ReLU, 3x3 conv, ReLU, 1x1 conv), all having 256 hidden units.
    """

    def __init__(self, in_channels, n_hidden, n_residual):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_hidden // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            in_channels=n_hidden // 2, 
            out_channels=n_hidden, 
            kernel_size=4, 
            stride=2, 
            padding=1
        )

        self.residual1 = Residual(n_hidden, n_hidden, n_residual)
        self.residual2 = Residual(n_hidden, n_hidden, n_residual)

        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, out):
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.residual1(out)
        out = self.residual2(out)
        out = self.relu2(out)
        return out
    
class Decoder(nn.Module):
    """
    Decoder
    """
    def __init__(self, in_channels, n_hidden, n_residual):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_hidden,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.residual1 = Residual(n_hidden, n_hidden, n_residual)
        self.residual2 = Residual(n_hidden, n_hidden, n_residual)

        self.relu = nn.ReLU()

        self.transpose1 = nn.ConvTranspose2d(
            in_channels=n_hidden,
            out_channels=n_hidden // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        
        self.transpose2 = nn.ConvTranspose2d(
            in_channels=n_hidden // 2,
            out_channels=1,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, out):
        out = self.conv1(out)
        out = self.residual1(out)
        out = self.residual2(out)
        out = self.relu(out)
        out = self.transpose1(out)
        out = self.transpose2(out)
        return out

class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_embeddings : number of embeddings
    - dim_embeddings : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_embeddings, dim_embeddings, beta):
        super(VectorQuantizer, self).__init__()
        self.n_embeddings = n_embeddings
        self.dim_embeddings = dim_embeddings
        self.beta = beta

        self.embedding = nn.Embedding(self.n_embeddings, self.dim_embeddings)
        self.embedding.weight.data.uniform_(-1.0 / self.n_embeddings, 1.0 / self.n_embeddings)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.dim_embeddings)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_embeddingsncoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_embeddingsncodings = torch.zeros(
            min_embeddingsncoding_indices.shape[0], self.n_embeddings).to(device)
        min_embeddingsncodings.scatter_(1, min_embeddingsncoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_embeddingsncodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_embeddingsncodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_embeddingsncodings, min_embeddingsncoding_indices

class VQVAE(nn.Module):
    def __init__(self, n_hidden = 128, n_residual = 32, n_embeddings = 512, dim_embedding = 64, beta = 0.25):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(1, n_hidden, 
                                n_residual)
        self.conv = nn.Conv2d(in_channels=n_hidden, 
                out_channels=dim_embedding,
                kernel_size=1, 
                stride=1)
        
        self.quantizer = VectorQuantizer(n_embeddings, 
            dim_embedding,
            beta)
        
        self.decoder = Decoder(dim_embedding,
            n_hidden, 
            n_residual)

    def forward(self, x):
        x = self.encoder(x)
        
        x = self.conv(x)
        loss, x_q, perplexity, _ = self.quantizer(x)
        x_hat = self.decoder(x_q)

        return loss, x_hat, perplexity

class Generator(nn.Module):
    def __init__(self, latent_size):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)
