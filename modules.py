import torch
import torch.nn as nn
import torchvision
from parameters import *


class WeightedLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WeightedLinear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.scale = (2 / in_channels) ** 0.5
        self.bias = self.linear.bias
        self.linear.bias = None

        # Initializing linear layer parameters
        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.linear(x * self.scale) + self.bias


class ADaIN(nn.Module):
    def __init__(self, number_channels):
        super(ADaIN, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(number_channels)
        self.ys = WeightedLinear(number_channels, number_channels)
        self.yb = WeightedLinear(number_channels, number_channels)

    def forward(self, content, style):
        # Calculating mean and std from content and style tensors
        # Dim denoted as 2nd and 3rd dimension for height and width respectively
        content = self.instance_norm(content)

        style_mean = self.yb(style)
        style_std = self.ys(style)

        # Scale and shift normalized content
        content = content.permute(0, 2, 3, 1)
        stylized_content = style_std * content + style_mean

        return stylized_content


class MappingNetwork(nn.Module):
    def __init__(self, z_dimension=z_dim, number_layers=num_layers, hidden_dimension=hidden_dim):
        super(MappingNetwork, self).__init__()

        layers = []
        layers.append(PixelNorm())
        for i in range(number_layers):
            # Setting the first and last layer to be dimension z to include different input and hidden layer size
            if i == 0:
                layers.append(WeightedLinear(z_dimension, hidden_dimension))
                layers.append(nn.LeakyReLU(0.2))
            elif i == num_layers - 1:
                layers.append(WeightedLinear(hidden_dimension, z_dimension))
            else:
                layers.append(WeightedLinear(hidden_dimension, hidden_dimension))
                layers.append(nn.LeakyReLU(0.2))
        # Passing through each layer into a parameter of the sequential network
        self.mapping_layers = nn.Sequential(*layers)

    def forward(self, z):
        # Activate mapping defined in init
        style = self.mapping_layers(z)
        return style


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class ScalingFactor(nn.Module):
    def __init__(self, number_channels):
        super(ScalingFactor, self).__init__()
        self.scaling_factors = nn.Parameter(torch.zeros(number_channels))

    def forward(self, noise):
        noise = noise.permute(0, 2, 3, 1)
        scaled_noise = noise * self.scaling_factors
        scaled_noise = scaled_noise.permute(0, 3, 1, 2)
        return scaled_noise


class WeightedConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WeightedConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.scale = (2 / (in_channels * (3 ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # Initialize convelution layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class StyleGANGeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StyleGANGeneratorBlock, self).__init__()

        # Defining operations conveyed in paper
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = WeightedConv(in_channels, out_channels)
        self.noise_scaler1 = ScalingFactor(out_channels)
        self.adain1 = ADaIN(out_channels)
        self.leaky1 = nn.LeakyReLU(0.2)
        self.conv2 = WeightedConv(out_channels, out_channels)
        self.noise_scaler2 = ScalingFactor(out_channels)
        self.adain2 = ADaIN(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, w_vector, noise_vector):
        # Ensure w_vector has dimension batch_size x out_channels
        # Ensure noise_vector has dimension batch_size x out_channels x img_size*2 x img_size*2
        x = x.permute(0, 3, 1, 2)
        x = self.upsample(x)
        # x = x.permute(0, 2, 3, 1)
        x = self.conv1(x)
        factored_noise = self.noise_scaler1(noise_vector)
        x = x + factored_noise
        x = self.leaky1(x)
        x = self.adain1(x, w_vector)

        x = x.permute(0, 3, 1, 2)
        x = self.conv2(x)
        x = self.relu(x)
        factored_noise = self.noise_scaler2(noise_vector)
        x = x + factored_noise
        x = self.adain2(x, w_vector)
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
        noise = torch.randn(1, self.num_channels, self.resolution, self.resolution).to("cuda:0")
        x = self.constant_input
        x = x + self.initial_scaler(noise) # Potentially change to different noise for each iter
        w_vec = self.mapping_network(z_vec)
        x = self.initial_adain(x, w_vec)
        x = x.permute(0, 3, 1, 2)
        x = self.initial_conv(x)
        x = x.permute(0, 2, 3, 1)
        current_feature_map_size = 4
        for i, layer in enumerate(self.layers):
            current_feature_map_size *= 2
            noise = torch.randn(1, self.channels[i], current_feature_map_size, current_feature_map_size).to("cuda:0")
            w_vec = nn.functional.interpolate(w_vec, (1, self.channels[i]))
            x = layer(x, w_vec, noise)
        x = x.permute(0, 3, 1, 2)
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
