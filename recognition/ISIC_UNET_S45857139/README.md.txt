# Segmentation of the ISIC 2018 Dataset with the Improved UNET
## Developed by Thomas Witherspoon for COMP3710 (UQ)

A typical UNET algorithm is characterised by an encoding path that captures the context of an input image, and a decoding path that restores the spatial details, with skip 
connections between the two paths to improve performance. My implementation of the improved UNET (inspired by https://arxiv.org/abs/1802.10508v1) makes use of residual blocks 
instead of standard convolutions.The residual block consists of two convolution layers followed by instance normalization. The output is then added to the residual to form the final output.
The improved UNet also features a localization module for the decoding path, which consists of a 3x3 convolution followed by a 1x1 convolution which reduces the number of channels by half.



The task was to segment the ISIC dataset with the improved UNet, with all labels having a minimum dice similarity coefficient of 0.8 on the test set. The dataset consists of 
skin lesions as images, and corresponding segmented mask images. The algorithm will learn to accurately predict the segmented mask for an unseen skin lesion image.

In the dataset, the training set size is set to 80% of the dataset, whereas the validation set is set to the remaining 20% of the dataset. There is no seperate test set. Having
a larger training set size is beneficial because it results in better model generalization and reduces overfitting.

The algorithm was trained on the rangpur HPC at UQ for a total of 10 epochs. Here are the results.

Dependencies/Libraries:
	- PyTorch
	- Python
	- PIL
	- os

	