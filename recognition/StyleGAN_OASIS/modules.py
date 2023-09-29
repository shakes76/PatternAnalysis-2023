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
from torch import nn, optim
from torchvision import datasets, transforms
import torch.nn.functional as F

"""
Define an alpha scheduler:
    fade_epochs     np.array of size max_depth-1 (or more) for number of epochs per
                    depth level in which to fade with the previous depth level. Index
                    0 is depth 1.
    hold_epochs     np.array of the size max_depth-1 (or more) for number of epochs per
                    depth level in which alpha will be held a 1 before progresing
                    to the next depth. Index 0 is depth 1.
    max_depth       Int describing how deep to control the fading
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
    def __init__(self, fade_epochs, hold_epochs, max_depth, batch_sizes, data_size):
        # Convert epochs into total number of iterations, AKA steps.
        self.fade_steps = np.array([np.ceil(fade_epochs[i]*data_size/batch_sizes[i]) for i in range(len(batch_sizes))])
        self.hold_steps = np.array([np.ceil(hold_epochs[i]*data_size/batch_sizes[i]) for i in range(len(batch_sizes))])
        self.max_depth = max_depth
        # Alpha is just the inverse of the number of steps
        self.alpha_steps = 1/self.fade_steps
        self.depth = 1
        self.steps = 0
        self.alpha = 0
        self.is_fade = False
        
    """
    Signal the scheduler that an epoch iteration has finished. This will generally
    increment self.alpha, finish fading and hold, or switch to the next depth.
    """
    def step(self):

        self.steps += 1
        # Within a fade, so increment alpha
        if self.steps < self.fade_steps[self.depth-1]:
            self.alpha += self.alpha_steps[self.depth-1]
            self.is_fade = True
        # Fade has finished but need to keep holding at alpha = 1
        elif self.steps < self.fade_steps[self.depth-1] + self.hold_steps[self.depth-1]:
            self.alpha = 1
            self.is_fade = False
        # Fade and hold are finished, time to move to next depth
        elif self.depth < self.max_depth:
            self.depth += 1
            self.alpha = 0
            self.steps = 0
            self.is_fade = True
        else:
            self.depth = self.max_depth
            self.alpha = 1
    """
    Return the current alpha value
    """
    def alpha(self):
        return self.alpha
    
    """
    Return the current depth level
    """
    def depth(self):
        return self.depth
    
    """
    Return if fading is currently scheduled (that is, 0<alpha<1)
    """
    def is_fade(self):
        return self.is_fade




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
        HeLayer.__init__(self, nn.Linear(in_ch, out_ch, bias=True), bias_fill=0, f=0.01)