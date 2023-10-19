# Improved UNet with the ISIC Dataset
## Daniel Kasumagic - s4742286

##  Description
### Design Task
The convolutional neural network developed for Task 1 is the Improved UNet, which for the duration of the report is called IUNet. This specific implementation was designed both for RGB images of 256x256 pixels and binary masks images of the same size. IUNet contextualises the images with 2D convolutions and downsamples to 32x32 images, then localization and construction of the segmentation masks begin. 
This implementation was fitted specially for the ISIC2018 Dataset, which depicts lesions on the skin.
