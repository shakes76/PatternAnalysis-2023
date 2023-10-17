
import torch
import torch.nn.functional as F
import torch.nn as nn


#Class implementation of model source code

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()

        self.in_channels_encoder = in_channels
        self.out_channels_encoder = 64 
        self.out_channels = out_channels

        self.context_path = nn.ModuleList()
        self.segmentation_layers = nn.ModuleList()
        self.localization_path = nn.ModuleList()
        
        self.context_pathway(4)
        self.localization_pathway(4)
        self.segmentation_pathway(4)
        
    #Defining the context pathway
    def context_pathway(self, levels):
        for _ in range(levels):
            self.context_path.append(context_module(self.in_channels_encoder, self.out_channels_encoder))
            self.in_channels_encoder = self.out_channels_encoder
            self.out_channels_encoder *= 2
            self.context_path.append(nn.MaxPool3d(kernel_size=2, stride=2))
    
    
    #Defining the localization pathway
    def localization_pathway(self, levels):
        self.in_channels_decoder = self.out_channels_encoder // 2
        self.out_channels_decoder = self.in_channels_decoder

        for _ in range(levels):
            # Upsample
            self.localization_path.append(nn.ConvTranspose3d(self.in_channels_decoder, self.out_channels_decoder, kernel_size=2, stride=2))
            # Convolution to reduce feature maps
            self.localization_path.append(nn.Conv3d(self.in_channels_decoder, self.out_channels_decoder // 2, kernel_size=3, padding=1))
            self.in_channels_decoder = self.out_channels_decoder
            self.out_channels_decoder //= 2
    
    #Defining the segmentation pathway
    def segmentation_pathway(self, levels):
        for _ in range(levels):
            segmentation_layer = nn.Conv3d(self.out_channels_decoder, self.out_channels, kernel_size=1)
            self.segmentation_layers.append(segmentation_layer)       
    
    def forward(self, x):
        # Forward pass through the context pathway
        features = []
        for module in self.context_path:
            x = module(x)
            features.append(x)
        
        # Forward pass through the localization pathway
        for i, module in enumerate(self.localization_path):
            x = module(x)
            x = torch.cat((x, features[-(i+1)]), dim=1)
        
        # Perform segmentation at each level
        segmentations = []
        for i, layer in enumerate(self.segmentation_layers):
            segmentation = layer(x)
            segmentations.append(segmentation)
            
        return segmentations

def context_module(in_channels, out_channels):
        module = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.3),  # Dropout is applied here
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01)
        )
        return module

def localization_module(in_channels, out_channels):
        module = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv3d(out_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv3d(out_channels // 2, out_channels // 2, kernel_size=1),
            nn.LeakyReLU(0.01)
        )
        return module