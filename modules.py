import torch
import torch.nn as nn
import torchvision
from parameters import *


class ADaIN(nn.Module):
    def __init__(self, number_channels):
        super(ADaIN, self).__init__()
        self.ys = nn.Parameter(torch.ones(number_channels))
        self.yb = nn.Parameter(torch.zeros(number_channels))

    def forward(self, content, style):
        # Calculating mean and std from content and style tensors
        # Dim denoted as 2nd and 3rd dimension for height and width respectively
        content = content.permute(0, 2, 3, 1)
        content_mean = content.mean(dim=[1, 2], keepdim=True)
        content_std = content.std(dim=[1, 2], keepdim=True)

        style_mean = style
        style_std = style

        # Normalizing the content tensor using the statistics from content tensor
        normalized_content = self.ys * (content - content_mean) / content_std + self.yb

        # Scale and shift normalized content
        stylized_content = style_std * normalized_content + style_mean

        return stylized_content


class MappingNetwork(nn.Module):
    def __init__(self, z_dimension=z_dim, number_layers=num_layers, hidden_dimension=hidden_dim):
        super(MappingNetwork, self).__init__()

        layers = []
        for i in range(number_layers):
            # Setting the first and last layer to be dimension z to include different input and hidden layer size
            if i == 0:
                layers.append(nn.Linear(z_dimension, hidden_dimension))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dimension, z_dimension))
            else:
                layers.append(nn.Linear(hidden_dimension, hidden_dimension))
                layers.append(nn.LeakyReLU(0.2))
        # Passing through each layer into a parameter of the sequential network
        self.mapping_layers = nn.Sequential(*layers)

    def forward(self, z):
        # Activate mapping defined in init
        style = self.mapping_layers(z)
        return style


class ScalingFactor(nn.Module):
    def __init__(self, number_channels):
        super(ScalingFactor, self).__init__()
        self.scaling_factors = nn.Parameter(torch.ones(number_channels))

    def forward(self, noise):
        noise = noise.permute(0, 2, 3, 1)
        scaled_noise = noise * self.scaling_factors
        scaled_noise = scaled_noise.permute(0, 3, 1, 2)
        return scaled_noise


class StyleGANGeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StyleGANGeneratorBlock, self).__init__()

        # Defining operations conveyed in paper
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.noise_scaler = ScalingFactor(out_channels)
        self.adain = ADaIN(out_channels)

    def forward(self, x, w_vector, noise_vector):
        # Ensure w_vector has dimension batch_size x out_channels
        # Ensure noise_vector has dimension batch_size x out_channels x img_size*2 x img_size*2
        x = x.permute(0, 3, 1, 2)
        x = self.upsample(x)
        # x = x.permute(0, 2, 3, 1)
        x = self.conv(x)
        factored_noise = self.noise_scaler(noise_vector)
        x = x + factored_noise
        x = self.adain(x, w_vector)
        return x


class StyleGANGenerator(nn.Module):
    def __init__(self, latent_dim, number_of_channels, resolution):
        super(StyleGANGenerator, self).__init__()

        self.latent_dim = latent_dim
        self.num_channels = number_of_channels
        self.resolution = resolution

        # Initial constant
        self.constant_input = nn.Parameter(torch.randn(1, number_of_channels, resolution, resolution))
        self.initial_scaler = ScalingFactor(number_of_channels)
        self.initial_adain = ADaIN(number_of_channels)
        self.initial_conv = nn.Conv2d(number_of_channels, number_of_channels, kernel_size=3, padding=1)
        self.mapping_network = MappingNetwork()

        self.layers = nn.ModuleList()
        self.channels = []
        in_channels = number_of_channels
        for i, out_channels in enumerate([number_of_channels // 2, number_of_channels // 4, number_of_channels // 8, 3]):
            self.layers.append(
                StyleGANGeneratorBlock(in_channels, out_channels)
            )
            in_channels = out_channels
            self.channels.append(in_channels)

    def forward(self, z_vec):
        noise = torch.randn(1, self.num_channels, self.resolution, self.resolution)
        x = self.constant_input
        x = x + self.initial_scaler(noise) # Potentially change to different noise for each iter
        w_vec = self.mapping_network(z_vec)
        x = self.initial_adain(x, w_vec)
        x = x.permute(0, 3, 1, 2)
        x = self.initial_conv(x)
        x = x.permute(0, 2, 3, 1)
        feature_map_size = 4
        for i, layer in enumerate(self.layers):
            feature_map_size *= 2
            noise = torch.randn(1, self.channels[i], feature_map_size, feature_map_size)
            w_vec = nn.functional.interpolate(w_vec, (1, self.channels[i]))
            x = layer(x, w_vec, noise)
        return x

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(num_channels, feature_map_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(feature_map_size, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(feature_map_size * 2, feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(feature_map_size * 4, feature_map_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(feature_map_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
