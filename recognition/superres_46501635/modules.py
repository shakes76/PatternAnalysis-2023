"""
Abstract: existing methods of super-res are performed in high resolution (HR) space -> computationally complex
            ESPCN fixes this by extracting feature maps in low resolution (LR) space -> i.e. downsample first and extract features
            A sub-pixel convolution layer which learns an array of upscaling filters to upscale the final LR feature maps to HR output, what this does
            is effectively replacing the handcrafted bicubic filter in the super-res pipeline with more complex upscaling filters specifically 
            trained for each feature map while also reducing the computational complexity 
"""


