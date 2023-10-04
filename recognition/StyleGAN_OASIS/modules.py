# -*- coding: utf-8 -*-
"""
File: modules.py

Purpose: Contains the necessary components for the Style GAN model. This includes
        - Alpha Scheduler
        - Generator model including the mapping network and synthesis blocks
        - Discriminator model including convolution blocks

@author: Peter Beardsley
"""

import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import torch.nn.functional as F

"""
Define an alpha scheduler:
    fade_epochs     np.array of size max_depth-1 (or more) for number of epochs per
                    depth level in which to fade with the previous depth level. Index
                    0 is depth 1.
    batch_sizes     List of batch sizes per depth level
    data_size       Int of number of images in an epoch
    
Notes: This is a scheduler for controlling the fading factor alpha that will
       blend a new progressive GAN layer with the previous layer. Depth will
       always start at 1, since style GAN doesn't start training at the depth=0
       layer (4x4), but rather depth=1 (8x8).
       self.steps counts the number of iterations within an epoch over the
       required number of epochs.
"""
class AlphaScheduler():
    def __init__(self, fade_epochs, batch_sizes, data_size):
        # Convert epochs into total number of iterations, AKA steps.
        self.fade_steps = np.array([np.ceil(fade_epochs[i]*data_size/batch_sizes[i])+2 for i in range(len(batch_sizes))])
        # Alpha is just the inverse of the number of steps
        self.alpha_steps = 1/self.fade_steps
        self.depth = 1
        self.steps = 0
        self.alpha = self.alpha_steps[0]
        self.new_depth = False
        
    """
    Step the alpha after every batch iteration
    """
    def stepAlpha(self):
        self.alpha += self.alpha_steps[self.depth-1]
        self.alpha = min(self.alpha, 1)

    """
    Steh the depth after all epoch runs per depth
    """
    def stepDepth(self):
        self.depth += 1
        if self.depth-1 < len(self.alpha_steps):
            self.alpha = self.alpha_steps[self.depth-1]
        else:
            # When finished, just hold at 1
            self.alpha = 1




"""
Define a PyTorch module that can normalise using RMS

Notes: A very small epsilon=1e-8 is added prior to taking the square root
       to avoid a potential sqrt(0) error.
"""
class RMS(nn.Module):
    def __init__(self):
        super(RMS, self).__init__()
        
    def forward(self, x):
        return x / (((x**2).mean(dim=1, keepdim=True) + 1e-8).sqrt())
    
  
"""
Define a PyTorch module that can equalise a Linear/Conv module using
the He constant
     bias_fill  Set the module bias, typical values are 0 or 1
     f          Factor to scale the He constant, typical values are 1 or 0.01
"""
class HeLayer(nn.Module):
    
    def __init__(self, module, bias_fill, f=1.0):
        super(HeLayer, self).__init__()
        self.module = module
        self.module.bias.data.fill_(bias_fill)
        self.module.weight.data.normal_(0,1)
        self.module.weight.data /= f
        HeConst = (2.0/np.prod(module.weight.size()[1:]))**0.5
        self.weight = HeConst*f
           
    def forward(self, x):
        x = self.module(x)
        x *= self.weight
        return x

"""
Extend the HeLayer to define an equalised Conv2d module
"""
class Conv2dHe(HeLayer):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        HeLayer.__init__(self, nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=True), bias_fill=0)

"""
Extend the HeLayer to define an equalised Linear module
"""
class LinearHe(HeLayer):
    def __init__(self, in_ch, out_ch, f=1.0):
        HeLayer.__init__(self, nn.Linear(in_ch, out_ch, bias=True), bias_fill=0, f=1.0)

"""
Define a PyTorch module that concatenates the mean standard deviation to the
layer output. This is a technique that aims to improve control, diversity, and
regularisation.
"""
class ConcatStdDev(nn.Module):
    def __init__(self):
        super(ConcatStdDev, self).__init__()
    
    def forward(self, x):
        size = list(x.size())
        size[1] = 1
        
        std = torch.std(x, dim=0)
        mean = torch.mean(std)
        return torch.cat((x, mean.repeat(size)), dim=1)
    
    
"""
Define the StyleGAN Mapping Network as a PyTorch module. This aims to learn a
manifold of latent z, expressed as w.
     in_ch      The dimension of z
     out_ch     The dimension of w
     depth      The mapping network depth, usually set to size 8
"""
class MappingNetwork(nn.Module):
    def __init__(self, in_ch, out_ch, depth=8):
        super(MappingNetwork, self).__init__()
        self.mappingNetwork = nn.ModuleList()
        
        ch = in_ch
        for i in range(depth):
            self.mappingNetwork.append(LinearHe(ch, out_ch, f=1)) # Try f=0.01
            ch = out_ch
        
        self.relu = torch.nn.LeakyReLU(0.2)
    
    """
    Feed forward:
       x    The normalised output of latent z
    """ 
    def forward(self, x):
        for fc in self.mappingNetwork:
            x = self.relu(fc(x))
            
        return x
"""
Adaptive Instance Normalisation PyTorch module
     channels   Number of channels of the synthesis layer
     w_size     The size of the Mapping Network output, w
     
Notes: scale and bias are derived from w by using an equaliser linear layer. This
       is typically done by outputing twice the channels but I've opted for 
       two seperate linear modules instead for simplicity
"""
class AdaIN(nn.Module):
    def __init__(self, channels, w_size):
        super(AdaIN, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale = LinearHe(w_size, channels)
        self.style_bias = LinearHe(w_size, channels)

    """
    Feed forward:
        x   The signal to feed into AdaIN
        w   The manifold output
    """
    def forward(self, x, w):
        x = self.instance_norm(x)
        style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        return style_scale * x + style_bias
    
"""
The B module from the style-based generator architecture that connects the noise
to the AdaIN.
     channels   Number of channels of the synthesis layer
    
Notes: weights are initialised to zero.
"""
class B(nn.Module):
    def __init__(self, channels):
        super(B, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    """
    Feed forward:
        s       The signal shape that defines the noise shape
        device  The CPU/GPU device for generating the noise on
    """
    def forward(self, s, device):
        b = torch.randn((s[0], 1, s[2], s[3]), device=device)
        return self.weight * b
    
"""
The first synthesis block of the StyleGAN generator architecture. This is similar
to the following blocks except for the constant 4x4 input.
     ch_in  The number of channels for the constant, typically 512
     ch_out The number of channels to output from the block
     w_size     The size of the Mapping Network output, w
"""
class SynthesisInitialBlock(nn.Module):
    def __init__(self, ch_in, ch_out, w_size):
        super(SynthesisInitialBlock, self).__init__()
        self.activate = nn.LeakyReLU(0.2, inplace=True)
        self.const = nn.Parameter(torch.ones((1, ch_in, 4, 4)))
        self.B1 = B(ch_out)
        self.adaIN1 = AdaIN(ch_out, w_size)
        self.conv2 = Conv2dHe(ch_out, ch_out)
        self.B2 = B(ch_out)
        self.adaIN2 = AdaIN(ch_out, w_size)
 
    """
    Feed Forward:
        a   The manifold output w
    
    Note: x is generated from a constant 4x4
    """
    def forward(self, a):
        x = self.const
        
        b1 = self.B1(torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]), device=a.device))
        b2 = self.B2(torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]), device=a.device))
        
        x = x + b1
        x = self.adaIN1(x, a)
        x = x + b2
        x = self.adaIN2(x, a)
        x = self.activate(x)
        return x

"""
The scaling synthesis blocks of the StyleGAN generator architecture. 
     ch_in  The number of channels for the previous block
     ch_out The number of channels to output from the block
     w_size     The size of the Mapping Network output, w
"""
class SynthesisBlock(nn.Module):
    def __init__(self, ch_in, ch_out, w_size):
        super(SynthesisBlock, self).__init__()
        self.activate = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = Conv2dHe(ch_in, ch_out)
        self.B1 = B(ch_out)
        self.adaIN1 = AdaIN(ch_out, w_size)
        self.conv2 = Conv2dHe(ch_out, ch_out)
        self.B2 = B(ch_out)
        self.adaIN2 = AdaIN(ch_out, w_size)

    """
    Feed Forward:
        x   The output of the previous layer
        a   The manifold output w
    """
    def forward(self, x, a):
        b1 = self.B1(torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device))
        b2 = self.B2(torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device))
        
        x = self.conv1(x)
        x = self.activate(x + b1)
        x = self.adaIN1(x, a)
        x = self.activate(x + b2)
        x = self.adaIN2(x, a)

        return x
    
"""
A PyTorch module implementation of the style-based generator architecture. The
main components are:
    RMS(z) -> Mapping Network -> w
    Synthesis Network for layer depth d:
        [0]: Const -> + B(Noise) -> + AdaIN(w) -> Conv + B(Noise) -> + AdaIN(w)
        [d]: Upsample -> Conv + B(Noise) -> + AdaIN(w) -> Conv + B(Noise) -> + AdaIN(w)
    RGB Output [d]: Conv (channels to RGB channels)

Purpose: Generator an image from latent z

Parameters:
    z_size      Size of latent z
    w_size      Size of manifold w
    channels    np.array[d] for channels at layer d
    rgb_ch      Number of RGB channels (3 for this project, 1 for grayscale)
    alphaSched  The AlphaScheduler object for managing the progresive GAN fading
"""
class Generator(nn.Module):
    def __init__(self, z_size, w_size, channels, rgb_ch, alphaSched):
        super(Generator, self).__init__()
        self.alphaSched = alphaSched
        self.normalise = RMS()
        self.mappingNetwork = MappingNetwork(z_size, w_size)
        self.synthesisNetwork = nn.ModuleList()
        self.rgbOutput = nn.ModuleList()
        
        self.synthesisNetwork.append(SynthesisInitialBlock(channels[0], channels[0], w_size))
        self.rgbOutput.append(Conv2dHe(channels[0], rgb_ch, kernel_size=1, stride=1, padding=0))
        
        for d in range(len(channels) - 1):
            ch_in = int(channels[d])
            ch_out = int(channels[d + 1])
            self.synthesisNetwork.append(SynthesisBlock(ch_in, ch_out, w_size))
            self.rgbOutput.append(Conv2dHe(ch_out, rgb_ch, kernel_size=1, stride=1, padding=0))
            
    def forward(self, z):
        # Normalise latent z, then pass to the mapping network
        w = self.mappingNetwork(self.normalise(z))

        # Scale depth based on the current AlphaScheduler depth
        for depth in range(self.alphaSched.depth+1):
            # First depth must be the SynthesisInitialBlock
            if isinstance(self.synthesisNetwork[depth], SynthesisInitialBlock):
                out = self.synthesisNetwork[0](w)
            # Subsequent blocks of SynthesisBlocks
            else:
                upsample = F.interpolate(out, scale_factor=2, mode="bilinear")
                out = self.synthesisNetwork[depth](upsample, w)
        
        # Transform from the current depth to RGB
        rgb_out = self.rgbOutput[self.alphaSched.depth](out)        
        
        # If fading, combine with the upscaled RGB output of the previous depth
        if self.alphaSched.is_fade:
            rgb_out_previous = self.rgbOutput[self.alphaSched.depth - 1](upsample)
            rgb_out = self.alphaSched.alpha * rgb_out + (1 - self.alphaSched.alpha) * rgb_out_previous
        
        # Map to a pixel value
        return torch.tanh(rgb_out)
    

"""
Scaling Convolutional block of the StyleGAN discriminator, which is composed of
two equalised convolution blocks
"""
class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv1 = Conv2dHe(ch_in, ch_out)
        self.conv2 = Conv2dHe(ch_out, ch_out)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.leaky(self.conv2(x))
        return x

 
"""
A PyTorch module implementation of the style-based discriminator architecture. The
main components are that from a progressive GAN.

Purpose: Classify an image image as being real or fake

Parameters:
    rgb_ch      Number of RGB channels (3 for this project, 1 for grayscale)
    channels    np.array[d] for channels at layer d
    alphaSched  The AlphaScheduler object for managing the progresive GAN fading
"""   
class Discriminator(nn.Module):
    def __init__(self, rgb_ch, channels, alphaSched):
        super(Discriminator, self).__init__()
        ch_in = channels[0]
        self.alphaSched = alphaSched
        self.synthesis_network = nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)
        self.stddev = ConcatStdDev()
        self.rgbInput = nn.ModuleList()

        self.rgbInput.append(Conv2dHe(rgb_ch, ch_in, kernel_size=1, stride=1, padding=0))
        for i in range(len(channels) - 1, 0, -1):
            conv_in = int(channels[i])
            conv_out = int(channels[i - 1])
            self.synthesis_network.append(ConvBlock(conv_in, conv_out))
            self.rgbInput.append(Conv2dHe(rgb_ch, conv_in, kernel_size=1, stride=1, padding=0))

        
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)  

        self.classifier = nn.Sequential(
            Conv2dHe(ch_in + 1, ch_in, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            Conv2dHe(ch_in, ch_in, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            Conv2dHe(ch_in, 1, kernel_size=1, padding=0, stride=1 ),
            nn.Sigmoid()
        )

    """
    Feed forward:
        x   The image data, either real or fake.
    """
    def forward(self, x):
        depth = len(self.synthesis_network) - self.alphaSched.depth

        out = self.leaky(self.rgbInput[depth](x))

        if self.alphaSched.depth == 0:
            out = self.stddev(out)
            return self.classifier(out).view(out.shape[0], -1)

        downscaled = self.leaky(self.rgbInput[depth + 1](self.avg_pool(x)))
        out = self.avg_pool(self.synthesis_network[depth](out))

        out = self.alphaSched.alpha * out + (1 - self.alphaSched.alpha) * downscaled

        for step in range(depth + 1, len(self.synthesis_network)):
            out = self.synthesis_network[step](out)
            out = self.avg_pool(out)

        out = self.stddev(out)
        
        return self.classifier(out).view(out.shape[0], -1)
    