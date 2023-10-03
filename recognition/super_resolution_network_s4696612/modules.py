import torch.nn as nn
import torch.nn.functional as F
6
class SuperResolution(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, 7, 1, 3)
        self.conv2 = nn.Conv2d(128, 128, 5, 1, 2)
        self.conv3 = nn.Conv2d(128, 64, 5, 1, 2)
        self.conv4 = nn.Conv2d(64, 4 ** 2, 3, 1, 1)
        self.pixel = nn.PixelShuffle(4)
        
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel(self.conv4(x))
        return x