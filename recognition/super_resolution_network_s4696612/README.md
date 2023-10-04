# Brain MRI Super-Resolution CNN
## COMP3710 Report - Pattern Recognition - Jarrod Mann - s4696612

## Introduction
The project aimed to create a deep learning model that could sufficiently upsample a low
resolution image. Specifically, the project focused on upsampling brain MRI scans. Creating an
effective model for this task would mean less overall storage space would be required for the
scans while they were not actively being used. Instead, a low resolution image could be stored,
and then be processed through the model each time its use was required. Therefore, the model
aims to reconstruct brain MRI scan images to as high a detail as possible.

## Model Implementation
An efficent sub-pixel convolutional neural network was implemented to complete the project.
This model consists of multiple normal convolutional layers, (activated with the rectified
linear unit function), and a pixelshuffle operation. In this model, the convolutions are
applied to the low resolution image before any upsampling is performed. The convolutions
generate a number of channels equal to the square of the upscaling factor; 16 filters are
made. The pixelshuffle opperation then 'shuffles' the components of these channels into 1
channel, thus creating a high resolution image. Through training, the convolutional layers
learn to give the pixelshuffle result channels that accurately represent the original image,
and the pixelshuffle operation learns the way the data should be arranged to successfully
recreated the high resolution image. In this way, super-resolution is achieved.