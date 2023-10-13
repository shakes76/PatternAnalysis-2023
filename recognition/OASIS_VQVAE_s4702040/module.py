import torch
from torch import nn
from torch.nn import functional as F

class ImprovedUNet(nn.module):
    def __init__(self, in_channels, n_filter):
        """
        Encoder
        """
        # Level 1 context pathway
        self.convE11 = nn.Conv2d(in_channels, n_filter, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        # Level 1 context module
        self.convE12 = nn.Conv2d(n_filter, n_filter, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.dropE1 = nn.Dropout3d(0.3)
        self.convE13 = nn.Conv2d(n_filter, n_filter, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.batchnormE1 = nn.BatchNorm2d(n_filter)

        # Level 2 context pathway
        self.convE21 = nn.Conv2d(n_filter, n_filter*2, kernel_size=3, 
                               stride=2, padding=1, bias=False)
        # Level 2 context module
        self.convE22 = nn.Conv2d(n_filter*2, n_filter*2, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.dropE2 = nn.Dropout3d(0.3)
        self.convE23 = nn.Conv2d(n_filter*2, n_filter*2, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.batchnormE1 = nn.BatchNorm2d(n_filter*2)

        # Level 3 context pathway
        self.convE31 = nn.Conv2d(n_filter*2, n_filter*4, kernel_size=3, 
                               stride=2, padding=1, bias=False)
        # Level 3 context module
        self.convE32 = nn.Conv2d(n_filter*4, n_filter*4, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.dropE3 = nn.Dropout3d(0.3)
        self.convE33 = nn.Conv2d(n_filter*4, n_filter*4, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.batchnormE1 = nn.BatchNorm2d(n_filter*4)
        
        # Level 4 context pathway
        self.convE41 = nn.Conv2d(n_filter*4, n_filter*8, kernel_size=3, 
                               stride=2, padding=1, bias=False)
        # Level 4 context module
        self.convE42 = nn.Conv2d(n_filter*8, n_filter*8, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.dropE4 = nn.Dropout3d(0.3)
        self.convE43 = nn.Conv2d(n_filter*8, n_filter*8, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.batchnormE5 = nn.BatchNorm2d(n_filter*8)
        
        # Level 5 context pathway
        self.convE51 = nn.Conv2d(n_filter*8, n_filter*16, kernel_size=3, 
                               stride=2, padding=1, bias=False)
        # Level 5 context module
        self.convE52 = nn.Conv2d(n_filter*16, n_filter*16, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.dropE5 = nn.Dropout3d(0.3)
        self.convE53 = nn.Conv2d(n_filter*16, n_filter*16, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.batchnormE5 = nn.BatchNorm2d(n_filter*16)

        """
        Decoder
        """
        # Level 1 localization pathway
        # Level 2 localization pathway
        # Level 3 localization pathway
        # Level 4 localization pathway

    def forward(self, x):
        pass
