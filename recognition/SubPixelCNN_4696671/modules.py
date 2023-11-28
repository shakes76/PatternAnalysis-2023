"""
DEFINITION OF THE MODEL
"""
# Imports
import torch.nn as nn

# Model Definition
class ESPCN(nn.Module):

    def __init__(self, in_channels, upscaling_factor=4):
        super(ESPCN, self).__init__()

        # From the paper, Tanh is their activation function.
        self.activation = nn.Tanh()

        # Paddings are selected to ensure that output image is the same size as input for all hidden conv layers.
        # Kernel sizes selected from Figure 1 in paper.
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        # need to end up with r^2 filters, where r is the upscaling factor
        self.conv4 = nn.Conv2d(64, in_channels * (upscaling_factor ** 2), kernel_size=3, padding=1)

        # Pixel shuffle is the sub-pixel layer, that takes the learnt feature maps and rearragnes them into the output, high resolution image.
        self.out = lambda x: nn.functional.pixel_shuffle(x, upscaling_factor)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.activation(x)

        x = self.conv4(x)

        x = self.out(x)
        return x
