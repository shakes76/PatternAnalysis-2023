# Improved UNet for ISIC 2018 Dataset
## Description
This project implements the UNet architecture as described in the paper: https://arxiv.org/abs/1802.10508v1
The implemented UNet was used on the ISIC 2018 dataset for segementation.
The difference between the Improved UNet and a basic UNet is...
The following is the figure of the Improved UNet architecture and a typical UNet architecture.

PICTURE OF Improved UNet
https://arxiv.org/pdf/1802.10508v1.pdf

PICTURE OF Basic UNet
https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5

## Dependencies
- torch 2.1.0+cu118 (used for implementing and training model)
- torchvision (used for transforms to preprocess data)
- PIL (used for loading images)
- matplotlib (used for plotting loss and output images)
- numpy (used for converting outputs to numpy arrays for plotting as image)

## Example Inputs/Outputs/Plots
Input:
Output:
Training Loss Plot:
3. provide example inputs, outputs and plots of your algorithm

## Pre-processing
Calculated the means and standard deviation for both the training set and the testing set images.

The training/validation/testing split is already done by the source and so nothing further was done with regards to that.
