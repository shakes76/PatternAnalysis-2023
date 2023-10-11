import torch.nn as nn
    
class ESPCN(nn.Module):
    """
    Implementation adapted to PyTorch from Tensorflow from https://keras.io/examples/vision/super_resolution_sub_pixel/
    """
    def __init__(self, upscale_factor=4, channels=1):
        super(ESPCN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=5, padding=2, padding_mode='reflect')
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv4 = nn.Conv2d(32, channels * (upscale_factor ** 2), kernel_size=3, padding=1, padding_mode='reflect')
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x