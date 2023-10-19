import torch
import torch.nn as nn
import torch.nn.functional as F

FLAT = -1
STRIDES = 2
PCNN_STRIDES = 1
KERN_SIZE = 3
CONV1_KERN_SIZE = 1
CONV_W_FACTOR = 32
FILTER_FACTOR = 2
KERN_FACTOR = 2
NO_FILTERS = 128
PCNN_IN_KERN_SIZE = 7
PCNN_OUT_KERN_SIZE = 1
PCNN_MID_KERN_SIZE = 1
KERN_INIT = 1.0
RESIDUAL_BLOCKS = 2
ENC_IN_SHAPE = (1, 80, 80)
NO_RESID_BLOCKS = 2
NO_PCNN_LAYERS = 4

class VectorQuantizer(nn.Module):
    def __init__(self, n_embeds, embed_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.embed_dim = embed_dim
        self.n_embeds = n_embeds
        self.beta = beta

        self.embeddings = nn.Embedding(n_embeds, embed_dim)

    def forward(self, x):
        in_shape = x.shape
        x = x.view(FLAT, self.embed_dim)
        distances = torch.sum(x**2, dim=1, keepdim=True) + torch.sum(self.embeddings.weight**2, dim=1) - 2 * torch.matmul(x, self.embeddings.weight.t())
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embeddings(encoding_indices).view(in_shape)
        commitment_loss = self.beta * torch.mean((quantized.detach() - x) ** 2)
        quantized = x + (quantized - x).detach()

        return quantized, commitment_loss

class Encoder(nn.Module):
    def __init__(self, lat_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, CONV_W_FACTOR, KERN_SIZE, stride=STRIDES, padding=1)
        self.conv2 = nn.Conv2d(CONV_W_FACTOR, 2 * CONV_W_FACTOR, KERN_SIZE, stride=STRIDES, padding=1)
        self.conv3 = nn.Conv2d(2 * CONV_W_FACTOR, 4 * CONV_W_FACTOR, KERN_SIZE, stride=STRIDES, padding=1)
        self.conv4 = nn.Conv2d(4 * CONV_W_FACTOR, lat_dim, 1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)

        return x

class Decoder(nn.Module):
    def __init__(self, lat_dim):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(lat_dim, 4 * CONV_W_FACTOR, KERN_SIZE, stride=STRIDES, padding=1)
        self.conv2 = nn.ConvTranspose2d(4 * CONV_W_FACTOR, 2 * CONV_W_FACTOR, KERN_SIZE, stride=STRIDES, padding=1)
        self.conv3 = nn.ConvTranspose2d(2 * CONV_W_FACTOR, 1, KERN_SIZE, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        return x

class VQVAE(nn.Module):
    def __init__(self, lat_dim, embeds, beta):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(lat_dim)
        self.vector_quantizer = VectorQuantizer(embeds, lat_dim, beta)
        self.decoder = Decoder(lat_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        quantized, commitment_loss = self.vector_quantizer(encoded)
        reconstructed = self.decoder(quantized)

        return reconstructed, commitment_loss

class Trainer(nn.Module):
    def __init__(self, train_vnce, lat_dim, n_embeds, beta):
        super(Trainer, self).__init__()
        self.train_vnce = train_vnce
        self.vqvae = VQVAE(lat_dim, n_embeds, beta)
        self.tot_loss = nn.MSELoss()
        self.recon_loss = nn.MSELoss()
        self.vq_loss = nn.MSELoss()

    def forward(self, x):
        recons, commitment_loss = self.vqvae(x)
        recon_loss = self.recon_loss(x, recons)
        total_loss = recon_loss + commitment_loss

        return total_loss

def build_vqvae(lat_dim, embeds, beta):
    vqvae = VQVAE(lat_dim, embeds, beta)
    return vqvae
