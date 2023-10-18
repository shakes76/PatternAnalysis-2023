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
- python 3.10.6
- torch 2.1.0+cu118 (used for implementing and training model)
- torchvision 0.16.0+cu118 (used for transforms to preprocess data)
- PIL/Pillow 9.5.0 (used for loading images)
- matplotlib 3.7.2 (used for plotting loss and output images)
- numpy 1.24.3 (used for converting outputs to numpy arrays for plotting as image)

## Example Usage
Show predict.py

## Example Inputs/Outputs/Plots
Input/Output
Training/Validation Loss Plot:
![Training and Validation Loss Plot](training_and_validation_loss.png)


## Pre-processing
The images were resized to a height and width of 1024 by 672 in order to standardise the pixel dimensions of the data set.
The training/validation/testing split is already done by the source and so nothing further was done with regards to that.

## References
