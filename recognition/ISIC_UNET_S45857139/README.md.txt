# Segmentation of the ISIC 2018 Dataset with the Improved UNET
## Developed by Thomas Witherspoon for COMP3710 (UQ)

A typical UNET algorithm is characterised by an encoding path that captures the context of an input image, and a decoding path that restores the spatial details, with skip 
connections between the two paths to improve performance. My implementation of the improved UNET (inspired by https://arxiv.org/abs/1802.10508v1) makes use of residual blocks 
instead of standard convolutions, and a localization module for decoding, which consists of a combination of a 3x3 and a 1x1 convolution. It also uses a dropout, and a leaky 
ReLU activation rather than a standard ReLU.

The task was to segment the ISIC dataset with the improved UNet, with all labels having a minimum dice similarity coefficient of 0.8 on the test set. The dataset consists of 
skin lesions as images, and then segmented mask images. The 






- Dependencies required
	- versions
	- reproducibiltiy of results

- Example outputs 
- Plots

