import torch as t
import torch.nn as nn
import torch.nn.functional as F

"""========== START UTILS =========="""

def conv(ch_in, ch_out, k_size, stride=2, pad=1, bn=True):
    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    return nn.Sequential(*layers)


def deconv(ch_in, ch_out, k_size, stride=2, pad=1, bn=True):
    layers = []
    layers.append(nn.ConvTranspose2d(ch_in, ch_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    return nn.Sequential(*layers)


class Residual(nn.Module):
    # Structure taken from Section 4.1 of Neural Discrete Representation Learning
    def __init__(self, ch_in, ch_out_inter, ch_out_final):
        super(Residual, self).__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(ch_in, ch_out_inter, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(ch_out_inter, ch_out_final, 1, 1, bias=False)
        )

    def forward(self, x):
        return x + self.net(x)


"""========== END UTILS =========="""


"""
GAN discriminator

model taken from COMP3710 Lab Demo 2 Part 3 (GAN) by Luke Halberstadt
"""
class Discriminator(nn.Module):
    def __init__(self, image_size=64, conv_dim=128):
        super(Discriminator, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4)
        self.fc = conv(conv_dim * 8, 1, int(image_size / 16), 1, 0, False)

    def forward(self, x):  # if image_size is 64, output shape is below
        out = F.leaky_relu(self.conv1(x), 0.05)  # (?, 64, 32, 32)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 16, 16)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 8, 8)
        out = F.leaky_relu(self.conv4(out), 0.05)  # (?, 512, 4, 4)
        out = F.sigmoid(self.fc(out)).squeeze()
        return out


"""
GAN generator

model taken from COMP3710 Lab Demo 2 Part 3 (GAN) by Luke Halberstadt
"""
class Generator(nn.Module):
    def __init__(self, z_dim=100, image_size=64, conv_dim=128):
        super(Generator, self).__init__()
        self.fc = deconv(z_dim, conv_dim * 8, int(image_size / 16), 1, 0, bn=False)
        self.deconv1 = deconv(conv_dim * 8, conv_dim * 4, 4)
        self.deconv2 = deconv(conv_dim * 4, conv_dim * 2, 4)
        self.deconv3 = deconv(conv_dim * 2, conv_dim, 4)
        self.deconv4 = deconv(conv_dim, 3, 4, bn=False)

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.fc(z)
        out = F.leaky_relu(self.deconv1(out), 0.05)
        out = F.leaky_relu(self.deconv2(out), 0.05)
        out = F.leaky_relu(self.deconv3(out), 0.05)
        out = F.tanh(self.deconv4(out))
        return out


"""
VQVAE Encoder
"""
class Encoder(nn.Module):
    # Structure taken from Section 4.1 of Neural Discrete Representation Learning
    def __init__(self, ch_in, num_hidden, residual_inter):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, num_hidden//2, 4, 2, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_hidden//2, num_hidden, 4, 2, 1)
        self.residual1 = Residual(num_hidden, residual_inter, num_hidden)
        self.residual2 = Residual(num_hidden, residual_inter, num_hidden)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.residual1(x)
        x = self.residual2(x)
        return x


"""
VQVAE Decoder
"""
class Decoder(nn.Module):
    # Structure taken from Section 4.1 of Neural Discrete Representation Learning
    def __init__(self, ch_in, num_hidden, residual_inter):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, num_hidden, 3, 1, 1)
        self.residual1 = Residual(num_hidden, residual_inter, num_hidden)
        self.residual2 = Residual(num_hidden, residual_inter, num_hidden)
        self.transpose1 = nn.ConvTranspose2d(num_hidden, num_hidden // 2, 4, 2, 1)
        self.transpose2 = nn.ConvTranspose2d(num_hidden // 2, 3, 4, 2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.transpose1(x)
        x = self.transpose2(x)
        return x


"""
VQVAE Vector Quantizer
"""
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z):
        # reshape BCHW -> BHWC
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)

        # quantization objective, move the embedding vectors towards the encoder outputs
        # (z - e)^2 = z^2 + e^2 - 2 e * z
        obj = (t.sum(z_flattened ** 2, dim=1, keepdim=True)
            + t.sum(self.embedding.weight**2, dim=1)
            - 2 * t.matmul(z_flattened, self.embedding.weight.t()))

        # find closest encodings
        min_encoding_indices_training = t.argmin(obj, dim=1)
        min_encoding_indices = t.argmin(obj, dim=1).unsqueeze(1)
        min_encodings = t.zeros(min_encoding_indices.shape[0], self.num_embeddings, device=z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_quantized = t.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = (F.mse_loss(z_quantized.detach(), z)
               + self.beta * F.mse_loss(z_quantized, z.detach()))

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()

        # reshape back to original BHWC -> BCHW
        z_quantized = z_quantized.permute(0, 3, 1, 2).contiguous()

        return loss, z_quantized, min_encodings, min_encoding_indices_training

    def quantize(self, z):
        # find closest encoding
        encoding_indices = z.unsqueeze(1)
        print(encoding_indices.shape)
        encodings = t.zeros(encoding_indices.shape[0], self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)
        # get quantized latent vector
        quantized = t.matmul(encodings, self.embedding.weight).view(1, 64, 64, 64)
        # reshape BHWC -> BCHW
        return quantized.permute(0, 3, 1, 2).contiguous()


class VQVAE(nn.Module):
    def __init__(self, num_hiddens, residual_inter,
                 num_embeddings, embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()

        self.encoder = Encoder(3, num_hiddens, residual_inter)
        self.conv1 = nn.Conv2d(num_hiddens, embedding_dim, 1, 1)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, num_hiddens, residual_inter)

    def forward(self, x):
        z = self.encoder(x)
        z = self.conv1(z)
        loss, quantized, _, _ = self.vq(z)
        x_recon = self.decoder(quantized) # reconstructed image
        return loss, x_recon