# TASK 1: Improved UNet for Image Segmentation

This project used an Improved UNet model on a segmentation problem for the ISIC melanoma dataset. This architecture was presented by [Isensee et al.](https://arxiv.org/pdf/1802.10508v1.pdf) [1], as a better alternative to the standard UNet structure.

# Dataset
The [ISIC 2017/18](https://challenge.isic-archive.com/data/#2017) dataset is a collection of skin lesion images, with associated segmentations and diagnoses. It is already split into training, validation and test subsets. A custom Dataset class was created to handle these images; it takes the image directory, matches the images to their corresponding mask and then transforms them. In this instance, the images were resized to (256,256), both due to memory requirements as well as simplicity in model (since it is a power of 2).

# Dependencies
A `conda` environment was created with the following packages:
```
pytorch
torchvision
matplotlib
pandas
```

# Model
The Improved UNet model was constructed as outlined by [1], diagram given below:
![HELLO](modelstructure.png)

This required the development of several additional blocks.

### Context
This context is described as a pre-activation residual block, basically two convolutions with a dropout layer inbetween.

### Localisation
This localisation block transfers encoded low spatial resolution information into a higher spatial resolution. In essence, it recombines the upsampled features with the saved outputs from the corresponding context block.

### Segmentation
Segmentation occurs at multiple levels, reducing the feature map back into one feature. This improves the accuracy of the model, as opposed to only performing segmentation on the final output.


# References
[1] Isensee, F., Kickingereder, P., Wick, W., Bendszus, M., and Maier-Hein, K. "Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge". Feb 28, 2018. Accessed from <https://arxiv.org/pdf/1802.10508v1.pdf>