"""
Model architecture for the super resolution network.
"""
import torch.nn as nn

# Super resolution model
class SuperResolutionModel(nn.Module):
    def __init__(self, upscale_factor=4, channels=3):
        super(SuperResolutionModel, self).__init__()
        
        self.inputs = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, channels * (upscale_factor ** 2), kernel_size=3, padding=1),
        )
        # PixelShuffle input shape:     (*, C x r^2, H, W)
        # PixelShuffle output shape:    (*, C, H x r, W x r)
        # This is responsible for outputting the upscaled image
        self.outputs = nn.Sequential(
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x):
        x = self.inputs(x)
        x = self.outputs(x)
        return x