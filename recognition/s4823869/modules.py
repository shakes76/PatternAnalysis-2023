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

class VectorQuantizer(nn.Module):
    def __init__(self, n_embeds, embed_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.embed_dim = embed_dim
        self.n_embeds = n_embeds
        self.beta = beta
        self.embeddings = nn.Parameter(torch.rand(embed_dim, n_embeds))

    def code_indices(self, flat):
        sim = torch.matmul(flat, self.embeddings)
        dists = torch.sum(flat ** 2, dim=1, keepdim=True) + torch.sum(self.embeddings ** 2, dim=0) - 2 * sim
        encodes = torch.argmin(dists, dim=1)
        return encodes

    def forward(self, x):
        in_shape = x.shape
        flat = x.view(FLAT, self.embed_dim)
        enc_ind = self.code_indices(flat)
        enc = F.one_hot(enc_ind, self.n_embeds).float()
        qtised = torch.matmul(enc, self.embeddings.t())
        qtised = qtised.view(in_shape)
        commit = self.beta * torch.mean((qtised.detach() - x) ** 2)
        codebook = torch.mean((qtised - x.detach()) ** 2)
        loss = codebook + commit
        return qtised, loss

class Encoder(nn.Module):
    def __init__(self, lat_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, CONV_W_FACTOR, KERN_SIZE, stride=STRIDES, padding=0),  # Set in_channels to 1
            nn.ReLU(),
            nn.Conv2d(CONV_W_FACTOR, 2 * CONV_W_FACTOR, KERN_SIZE, stride=STRIDES, padding=0),
            nn.ReLU(),
            nn.Conv2d(2 * CONV_W_FACTOR, 4 * CONV_W_FACTOR, KERN_SIZE, stride=STRIDES, padding=0),
            nn.Conv2d(4 * CONV_W_FACTOR, lat_dim, 1, padding=0)
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, lat_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(lat_dim, 4 * CONV_W_FACTOR, KERN_SIZE, stride=STRIDES, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4 * CONV_W_FACTOR, 2 * CONV_W_FACTOR, KERN_SIZE, stride=STRIDES, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * CONV_W_FACTOR, 1, KERN_SIZE, padding=1)
        )

    def forward(self, x):
        return self.decoder(x)


class VQVAE(nn.Module):
    def __init__(self, lat_dim, embeds, beta):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(lat_dim)
        self.vector_quantizer = VectorQuantizer(embeds, lat_dim, beta)
        self.decoder = Decoder(lat_dim)

    def forward(self, x):
        enc_out = self.encoder(x)
        qtised, vq_loss = self.vector_quantizer(enc_out)
        recon = self.decoder(qtised)
        return recon, vq_loss

class Trainer(nn.Module):
    def __init__(self, train_vnce, lat_dim, n_embeds, beta):
        super(Trainer, self).__init__()
        self.train_vnce = train_vnce
        self.lat_dim = lat_dim
        self.n_embeds = n_embeds
        self.vqvae = VQVAE(lat_dim, n_embeds, beta)
        self.optimizer = torch.optim.Adam(self.vqvae.parameters())
        self.tot_loss = 0
        self.recon_loss = 0
        self.vq_loss = 0

    def forward(self, x):
        recons, vq_loss = self.vqvae(x)
        recon_loss = torch.mean((x - recons) ** 2) / self.train_vnce
        total_loss = recon_loss + vq_loss
        return total_loss, recon_loss, vq_loss

class PixelConvolution(nn.Module):
    def __init__(self, mask_type):
        super(PixelConvolution, self).__init__()
        self.mask_type = mask_type
        self.conv = nn.Conv2d(NO_FILTERS, NO_FILTERS, kernel_size=KERN_SIZE, padding=KERN_SIZE // 2)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        self.apply_mask()
        x = self.conv(inputs)
        x = self.relu(x)
        return x

    def apply_mask(self):
        kernel = self.conv.weight
        mask = torch.zeros_like(kernel)
        mask[: kernel.size(0) // KERN_FACTOR, ...] = KERN_INIT
        mask[kernel.size(0) // KERN_FACTOR, : kernel.size(1) // KERN_FACTOR, ...] = KERN_INIT
        if self.mask_type == "B":
            mask[kernel.size(0) // KERN_FACTOR, kernel.size(1) // KERN_FACTOR, ...] = KERN_INIT
        self.conv.weight.data = self.conv.weight.data * mask

class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.filters = filters
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=CONV1_KERN_SIZE, padding=CONV1_KERN_SIZE // 2)
        self.relu1 = nn.ReLU()
        self.pixel_conv = PixelConvolution(mask_type="B")
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=CONV1_KERN_SIZE, padding=CONV1_KERN_SIZE // 2)
        self.relu2 = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.relu1(x)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return inputs + x

class PixelCNN(nn.Module):
    def __init__(self):
        super(PixelCNN, self).__init__()
        self.pixel_conv_layers = nn.ModuleList([PixelConvolution(mask_type="A") for _ in range(NO_RESIDUAL_BLOCKS)])
        self.pcnn_layers = nn.ModuleList([PixelConvolution(mask_type="B") for _ in range(NO_PCNN_LAYERS)])
        self.out_conv = nn.Conv2d(NO_FILTERS, n_embeds, kernel_size=PCNN_OUT_KERN_SIZE, stride=PCNN_STRIDES)

    def forward(self, pcnn_ins):
        x = pcnn_ins.float()
        for layer in self.pixel_conv_layers:
            x = layer(x)
        for layer in self.pcnn_layers:
            x = layer(x)
        out = self.out_conv(x)
        return out

# You can now add data loading and training loop for your model.
