# Improved UNet with the ISIC Dataset
## Daniel Kasumagic - s4742286

##  Description
### Design Task
The convolutional neural network developed for Task 1 is the Improved UNet, which for the duration of the report is called IUNet. This specific implementation was designed both for RGB images of 256x256 pixels and binary masks images of the same size. The purpose of the localization pathway is to move contextual information encoded at a low spatial resolution to a higher spatial resolution through the application of features from lower layers of the network. To accomplish this, the low resolution feature maps are first upsampled using a straightforward upscale method that duplicates the feature voxels twice in each spatial dimension. This is followed by a 3x3 convolution that reduces the number of feature maps in half. This implementation was fitted specially for the ISIC2018 Dataset, which is an image dataset depicting lesions on the skin.


### Model Overview

### Aims and Objectives

## Implementation
### Encoder

### Decoder

## Dependencies
### Pytorch

### Dataset

### Hardware

## Results
### Data Preprocessing

### Example Inputs and Outputs

### Training Results (Epoch vs Loss)

## Reproducibility
### Hyperparameters

### Optimiser

### Problems and Potential Improvements

## References
