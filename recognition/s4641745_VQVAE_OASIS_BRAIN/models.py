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
    layers.append(nn.ConvTranspose2d(ch_in, ch_out, k_size, stride, pad))
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
            nn.Conv2d(ch_out_inter, ch_out_final, 1, 1, False)
        )

    def forward(self, x):
        return x + self.net(x)


"""========== END UTILS =========="""


"""
GAN discriminator

model taken from COMP3710 Lab Demo 2 Part 3 (GAN) by Luke Halberstadt
"""
class Discriminator(nn.Module):
    def __init__(self, image_size=128, conv_dim=64):
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
    def __init__(self, z_dim=256, image_size=128, conv_dim=64):
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