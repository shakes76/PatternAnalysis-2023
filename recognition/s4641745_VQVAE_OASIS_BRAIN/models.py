import torch as t
import torch.nn as nn
import torch.nn.functional as F

"""
GAN discriminator

model taken from COMP3710 Lab Demo 2 Part 3 (GAN) by Luke Halberstadt
"""
def conv(ch_in, ch_out, k_size, stride=2, pad=1, bn=True):
    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    return nn.Sequential(*layers)


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
def deconv(ch_in, ch_out, k_size, stride=2, pad=1, bn=True):
    layers = []
    layers.append(nn.ConvTranspose2d(ch_in, ch_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    return nn.Sequential(*layers)


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

"""