"""
Abstract: existing methods of super-res are performed in high resolution (HR) space -> computationally complex
            ESPCN fixes this by extracting feature maps in low resolution (LR) space -> i.e. downsample first and extract features
            A sub-pixel convolution layer which learns an array of upscaling filters to upscale the final LR feature maps to HR output, what this does
            is effectively replacing the handcrafted bicubic filter in the super-res pipeline with more complex upscaling filters specifically 
            trained for each feature map while also reducing the computational complexity 

Existing interpolation methods do not bring additional information into the already ill-posed reconstruction problem

In ESPCN, upscaling is handled by final layer of network, meaning each LR image is directly fed into the network and feature extraction occurs through
nonlinear convolutions in LR space. Due to reduced input res, a smaller filter size can be effectively used to integrate same information while 
maintaining a given contextual area. The resolution and filter size reduction lowers the computational cost substantially enough to allow super-res of 
HD videos in real-time.
For a network with L layers, we learn n(L - 1) upscaling filters for the n(L - 1) feature maps as opposed to one upscaling filter for the input image.
Not using of an explicit interpolation filter means that the network will implicitly learn the processing necessary for SR. Thus, the network is able
to learn a better and more complex LR to HR mapping compared to a single fixed filter upscaling at the first layer. This results in additional gains
in reconstruction accuracy!
"""

"""
tanh is used instead of relu
To synthesise low-resolution samples, HR images are blurred with gaussian filter (in this assignment, pytorch's resize is used instead (as requested))
Initial learning rate is set at 0.01 and final learning rate is 0.0001 and updated gradually when the improvement of the cost is smaller than a threshold
u. 
With this change of learning rate + size of dataset, it is more feasible to lower the number of epochs, looking between 5 - 20 epochs.
Peak signal to noise ratio (PSNR) is used as performance metric 

"""

import torch.nn as nn

class ESPCN(nn.Module):
    def __init__(self, upscale_factor, input_channels=1):
        # input channel 1 because only greyscale image 
        super(ESPCN, self).__init__()

        # Feature extraction!
        self.feature_extraction = nn.Sequential(
            # Spatial dimensions do not change in feature extraction phase
            # (batch_size, 1, H, W) 
            nn.Conv2d(input_channels, 64, kernel_size=5, padding=2),
            # (batch_size, 64, H, W)
            nn.Tanh(), # Tanh used to output values [-1,1] which is beneficial for image data, especially when normalised to same range.
                       # prevents saturation and vanishing gradient problems better than sigmoid or relu
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            # (batch_size, 32, H, W)
            nn.Tanh()
        )

        # Second part: upscaling using sub-pixel convolution
        self.sub_pixel = nn.Sequential(
            # Increases number of channels by a factor of (upscale_factor ** 2)
            nn.Conv2d(32, input_channels * (upscale_factor ** 2), kernel_size=3, padding=1),
            # (batch_size, input_channels * upscale_factor^2, H, W)
            # this operation reshapes the tensor by splitting the channels into 
            # (upscale_factor, upscale_factor) blocks and reshuffling them to increase the spatial dimensions.
            nn.PixelShuffle(upscale_factor),
            # (batch_size, input_channels, H * upscale_factor, W * upscale_factor)
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        # takes input tensor 'x' and passes it through feature extraction layers,
        # sub-pixel convolution layers to produce the super-resolved output
        x = self.sub_pixel(x)
        return x