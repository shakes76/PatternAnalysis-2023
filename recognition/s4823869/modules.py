"""
Generative Adversarial Network (GAN) Modules

This module defines the neural network modules for a GAN, including the Generator and Discriminator networks.
The code is organized into various classes, each representing a specific component of the GAN architecture.

@author: Yash Mittal
@ID: s48238690
"""

import torch
import torch.nn.functional as F
from torch import nn

# Scaling factors
factors = [1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]


class StyleMapping(nn.Module):
    """
    Style Mapping Network

    This module maps the latent noise vector (Z) to the intermediate style vector (W) that controls the
    generation of images by the Generator.
    """

    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            PixelwiseNormalization(),
            WeightScaledLinear(z_dim, w_dim),
            nn.ReLU(),
            WeightScaledLinear(w_dim, w_dim),
            nn.ReLU(),
            WeightScaledLinear(w_dim, w_dim),
            nn.ReLU(),
            WeightScaledLinear(w_dim, w_dim),
            nn.ReLU(),
            WeightScaledLinear(w_dim, w_dim),
            nn.ReLU(),
            WeightScaledLinear(w_dim, w_dim),
            nn.ReLU(),
            WeightScaledLinear(w_dim, w_dim),
            nn.ReLU(),
            WeightScaledLinear(w_dim, w_dim),
        )

    def forward(self, x):
        return self.mapping(x)


class PixelwiseNormalization(nn.Module):
    """
    Pixel-wise Normalization

    This module performs pixel-wise normalization on the input tensor.
    """

    def __init__(self):
        super(PixelwiseNormalization, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


# Untangled data distribution for easier generator training
# Z -> style factor W

class WeightScaledLinear(nn.Module):
    """
    Weight-Scaled Linear Layer

    This module is a linear (fully connected) layer with weights scaled for better training stability.
    """

    def __init__(self, in_features, out_features):
        super(WeightScaledLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = (2 / in_features) ** 0.5  # Scales to maintain the standard deviation close to 1
        self.bias = self.linear.bias
        self.linear.bias = None

        nn.init.normal_(self.linear.weight)  # Initialize weights from a normal distribution
        nn.init.zeros_(self.bias)  # Initialize biases to zero

    def forward(self, x):
        return self.linear(x * self.scale) + self.bias

# Adaptive Instance Normalization
# Embed style factor W into the layers of the generator
# Provides global feature control
class WeightScaledConv2d(nn.Module):
    """
    Weight-Scaled Conv2d Layer

    This module is a convolutional layer with weights scaled for better training stability.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(WeightScaledConv2d, self).__init__()  # Fixed super() call
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (2 / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # Initialize the convolutional layer
        nn.init.normal_(self.conv.weight)  # Initialize weights from a normal distribution
        nn.init.zeros_(self.bias)  # Initialize biases to zero

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


# Adaptive Instance Normalization for style transfer
class AdaptiveInstanceNorm(nn.Module):
    """
    Adaptive Instance Normalization

    This module performs adaptive instance normalization on the input tensor, allowing style transfer.
    """

    def __init__(self, channels, w_dim):
        super(AdaptiveInstanceNorm, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale = WeightScaledLinear(w_dim, channels)
        self.style_bias = WeightScaledLinear(w_dim, channels)

    def forward(self, x, w):
        # Style Scale
        style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)

        # Style Bias
        style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)

        # Apply Instance Normalization
        x = self.instance_norm(x)

        # Apply Style Scaling and Biasing
        return style_scale * x + style_bias

# Convolutional block for the generator
class ConvolutionBlock(nn.Module):
    """
    Convolutional Block

    This module represents a block of convolutional layers used in the generator.
    """

    def __init__(self, in_channels, out_channels):
        super(ConvolutionBlock, self).__init__()
        self.conv1 = WeightScaledConv2d(in_channels, out_channels)
        self.conv2 = WeightScaledConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.leaky(self.conv2(x))
        return x

# Noise injection module
class NoiseInjection(nn.Module):
    """
    Noise Injection Module

    This module injects noise into the input tensor.
    """

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        noise = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
        return x + self.weight + noise

# Generator block with style factor and specific noise
class GeneratorBlock(nn.Module):
    """
    Generator Block

    This module represents a block in the generator with adaptive instance normalization and noise injection.
    """

    def __init__(self, in_channel, out_channel, w_dim):
        super(GeneratorBlock, self).__init__()
        self.conv1 = WeightScaledConv2d(in_channel, out_channel)
        self.conv2 = WeightScaledConv2d(out_channel, out_channel)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.inject_noise1 = NoiseInjection(out_channel)
        self.inject_noise2 = NoiseInjection(out_channel)
        self.AdaptiveInstanceNorm1 = AdaptiveInstanceNorm(out_channel, w_dim)
        self.AdaptiveInstanceNorm2 = AdaptiveInstanceNorm(out_channel, w_dim)

    def forward(self, x, w):
        x = self.AdaptiveInstanceNorm1(self.leaky(self.inject_noise1(self.conv1(x))), w)
        x = self.AdaptiveInstanceNorm2(self.leaky(self.inject_noise2(self.conv2(x))), w)
        return x

# Generator class
class Generator(nn.Module):
    def __init__(self, z_dim, w_dim, in_channels, img_channels=3):
        super(Generator, self).__init__()
        self.initial_constant = nn.Parameter(torch.ones(1, in_channels, 4, 4))
        self.style_mapping = StyleMapping(z_dim, w_dim)
        self.initial_adain1 = AdaptiveInstanceNorm(in_channels, w_dim)
        self.initial_adain2 = AdaptiveInstanceNorm(in_channels, w_dim)
        self.initial_noise1 = NoiseInjection(in_channels)
        self.initial_noise2 = NoiseInjection(in_channels)
        self.initial_conv = WeightScaledConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)

        self.initial_rgb = WeightScaledConv2d(in_channels, img_channels, kernel_size=1, stride=1, padding=0)
        self.progressive_blocks, self.rgb_layers = (
            nn.ModuleList([]),
            nn.ModuleList([self.initial_rgb])
        )

        for i in range(len(factors) - 1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i + 1])
            self.progressive_blocks.append(GeneratorBlock(conv_in_c, conv_out_c, w_dim))
            self.rgb_layers.append(WeightScaledConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0))

    def fade_transition(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, noise, alpha, steps):
        w = self.style_mapping(noise)
        x = self.initial_adain1(self.initial_noise1(self.initial_constant), w)
        x = self.initial_conv(x)
        out = self.initial_adain2(self.leaky(self.initial_noise2(x)), w)

        if steps == 0:
            return self.initial_rgb(x)

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode='bilinear')
            out = self.progressive_blocks[step](upscaled, w)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)

        return self.fade_transition(alpha, final_upscaled, final_out)


# Discriminator class
class Discriminator(nn.Module):
    """
    Discriminator Network

    This module defines the discriminator network of the GAN for image discrimination.
    """

    def __init__(self, in_channels, img_channels=3):
        super(Discriminator, self).__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        for i in range(len(factors) - 1, 0, -1):
            conv_in = int(in_channels * factors[i])
            conv_out = int(in_channels * factors[i - 1])
            self.prog_blocks.append(ConvolutionBlock(conv_in, conv_out))
            self.rgb_layers.append(
                WeightScaledConv2d(img_channels, conv_in, kernel_size=1, stride=1, padding=0)
            )

        self.initial_rgb = WeightScaledConv2d(
            img_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.final_block = nn.Sequential(
            WeightScaledConv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WeightScaledConv2d(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WeightScaledConv2d(in_channels, 1, kernel_size=1, padding=0, stride=1),
        )

    def blend_images(self, alpha, downscaled, out):
        return alpha * out + (1 - alpha) * downscaled

    def compute_minibatch_std(self, x):
        batch_statistics = (torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3]))
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, alpha, steps):
        cur_step = len(self.prog_blocks) - steps
        out = self.leaky(self.rgb_layers[cur_step](x))

        if steps == 0:
            out = self.compute_minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.blend_images(alpha, downscaled, out)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.compute_minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)
