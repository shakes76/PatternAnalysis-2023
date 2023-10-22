# Improved U-Net For Image Segmentation
Author: Michael Hansen, s4428306

# Improved U-Net Algorithm
Improved U-Net is a deep learning convolutional network that segments an image into a number of classes. It does this with a modified version of the original U-Net architecture[^1] that adds segmentation layers to the localization pathway of the network[^2]. This implementation uses 2D convolutions and resizes images to 128x128.
The dataset used is the ISIC 2018 dataset.

# Problem Description
Image segmentation is used to find and classify objects in an image on a pixel by pixel basis. While some algorithms classify the image itself, or create a bounding box around the objects in it, segmentation attempts to define the exact location of the object in question, and can even be used to find the outline of the object.
This model is used to classify skin lesions in the ISIC 2018 dataset.

# Improved U-Net Architecture
The following figure, taken from [^2], outlines the large scale architecture of Improved U-Net. Note that the implementation of this model differs from [^2] in that all convolutions and upsamples are 2D, rather than 3D. Each module/layer has the number of output filters depicted in the diagram.
![IUNet diagram](/recognition/s4428306_UNET/IUNetArchitecture.png)
The model can be broken down into two pathways:
## The context pathway:
This pathway consists of context modules and 3x3 convolutions, and is responsible for high level feature extraction.
Each context module consists of the following layers:
- 3x3 2D convolution
- 3x3 2D convolution
- Dropout layer with p = 0.3
Every context module has a skip connection that adds its inputs to its outputs, and this is in turn concatenated with the input of the localization modules in the localization pathway to prevent vanishing gradients.
## The localization pathway:
This pathway consists of upsampling modules, localization modules, and segmentation layers.
Each upsampling module consists of the following layers:
- 2x2 upsampling layer
- 3x3 2D convolution
While each localization module consists of:
- A concatenation of the output from the context module symmetric to it
- 3x3 2D convolution
- 1x1 2D convolution
The segmentation layers are implemented as 1x1 2D convolutions with 1 output filter, as stated in [^3].
Finally, instead of a softmax layer, this implemenation uses a sigmoidal activation, as there is only one class being segmented: the lesions. All activation functions, save for the final layer, are Leaky ReLU with a slope of 0.01.

There are some other differences between this implementation and the one in [^2]:
- The model is trained with a dice loss function for one class.
- Padding was applied evenly to the edges of input in each convolution to ensure the inputs and outputs had the same size.
- The model was trained for 30 epochs with a batch size of 64.
- The Adam optimizer used had the default settings given by `tensorflow.keras.optimizers.Adam()`.
- No l2 weight decay was used.

# Example Outputs
The model produces pixelated outputs that roughly conform to the mask of the depicted lesion.
## Example 1
Image:
![Example 1 image](/recognition/s4428306_UNET/example1_image.png)
Mask:
![Example 1 mask](/recognition/s4428306_UNET/example1_mask.png)
Prediction:
![Example 1 prediction](/recognition/s4428306_UNET/example1_pred.png)
## Example 2
Image:
![Example 2 image](/recognition/s4428306_UNET/example2_image.png)
Mask:
![Example 2 mask](/recognition/s4428306_UNET/example2_mask.png)
Prediction:
![Example 2 prediction](/recognition/s4428306_UNET/example2_pred.png)

# Preprocessing
All images were resized to 128x128 using bicubic resampling. This means that aspect ratios were not preserved, and some images may have been squashed. A center crop of 256x256 was attempted, but this cut out too much data to be useful.
All images were passed to the model after having their colour values divided by 255 for normalization, and ground truth masks were converted to 0s and 1s, with 1s being where a lesion was present and 0s being where there was no lesion.
The data was split into a 1796 example training set, 400 example validation set, and a 400 example test set. This was to ensure that the model generalises well on the test set.

# Training Plots
These plots show the model training over 10 epochs.
## Training Loss
![Training loss](/recognition/s4428306_UNET/loss_plot.png)
## Training Accuracy
![Training accuracy](/recognition/s4428306_UNET/accuracy_plot.png)

# Testing
The model successfully achieves a dice accuracy of 0.79 on the test data set.

# Usage
This model is run using `python test_driver.py`. The model keras file, plots and example images are saved in:
/home/Student/s4428306/report/

# Dependencies
- python 3.11.5
- tensorflow 2.12.0
- keras 2.12.0
- numpy 1.23.5
- pillow 9.4.0
- matplotlib 3.7.2

# Apologies
Git commits were overloaded towards the end because I was trying to fix code while simultaneously training the model.

# References
[^1]: U-Net: Convolutional Networks for Biomedical Image Segmentation (arXiv:1505.04597)
[^2]: Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge (arXiv:1802.10508)
[^3]: CNN-based Segmentation of Medical Imaging Data (arXiv:1701.03056)

