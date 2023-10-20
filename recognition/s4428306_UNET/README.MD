# Improved U-Net For Image Segmentation
Author: Michael Hansen, s4428306

#Improved U-Net Algorithm
Improved U-Net is a deep learning convolutional network that segments an image into a number of classes. It does this with a modified version of the original U-Net architecture[^1] that adds segmentation layers to the localization pathway of the network[^2]. This implementation uses 2D convolutions and resizes images to 128x128.
The dataset used is the ISIC 2018 dataset.

#Problem Description
Image segmentation is used to find and classify objects in an image on a pixel by pixel basis. While some algorithms classify the image itself, or create a bounding box around the objects in it, segmentation attempts to define the exact location of the object in question, and can even be used to find the outline of the object.
This model is used to classify skin lesions in the ISIC 2018 dataset.

#Improved U-Net Architecture
The following figure, taken from [^2], outlines the large scale architecture of Improved U-Net.
DISPLAY UNET DIAGRAM FROM SECOND PAPER
![IUNet diagram](/recognition/s4428306_UNET/IUNetArchitecture.png)
DESCRIBE PATHWAYS
CONTEXT MODULES
SKIP CONNECTIONS AND CONCATS
UPSAMPLING MODULES
LOCALIZATION MODULES
SEGMENTATION LAYERS [^3]
SOFTMAX/SIGMOID
TRAINING PARAMETERS            NOTE DIFFERENCES BETWEEN PAPER AND IMPLEMENTATION THROUGHOUT

EXAMPLE OUTPUTS

PREPROCESSING

TRAINING PLOTS

WHAT OTHER THINGS NEEDED TO BE MENTIONED?
CHECK .py FILES
Where is model/images saved?
How to use test driver.

DEPENDENCIES
Use conda environment info and py files to get

#References
[^1]: Original U-Net paper HERE.
[^2]: Improved U-Net paper HERE.
[^3]: Second Improved U-Net paper HERE.

#Apologies
Git commits were overloaded towards the end because I was trying to fix code while simultaneously training the model.
