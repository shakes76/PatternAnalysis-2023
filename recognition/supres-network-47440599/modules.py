import torch
import torch.nn as nn

class SubPixel(nn.Module):
    def __init__(self, upscale_factor=4, channels=1):
        super(SubPixel, self).__init__()
        
        conv_args = {
            "padding": 1,
        }
        
        #Adding convolutional layers
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        #Pixel shuffle used to upscale the images
        self.upsample = nn.PixelShuffle(upscale_factor)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.conv4(x)
        x = self.upsample(x)
        return x
    
