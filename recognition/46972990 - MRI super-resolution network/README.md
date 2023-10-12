# Implementation of a Brain MRI Super-Resolution Network (ESPCN)
Benjamin Guy 46972990

## Introduction
In this recognition task, a super-resolution network model will be created that can up-scale by a factor of 4 on downsampled ADNI brain MRI scan images to produce a reasonably clear image. The dataset used for this task will be the [ADNI brain dataset](https://adni.loni.usc.edu/). The algorithm used to complete this recognition task is the [ESPCN (Efficient Sub-Pixel CNN)](https://keras.io/examples/vision/super_resolution_sub_pixel/). This algorithm reconstructs a high-resolution version of an image by leveraging
efficient sub-pixel convolution layers to learn image upscaling filters. In the case of this recognition task, the ESPCN model will take down-scaled images (by a factor of 4) from the ADNI brain dataset and attempt to up-scale the images back to their original resolution without any perceptual loss of quality. 

![Figure 1 displaying a comparison between the down-scaled image, up-scaled image by the model, and the original image.](images/Figure_1.png?raw=true "Model performance 1")
![Figure 2 displaying a comparison between the down-scaled image, up-scaled image by the model, and the original image.](images/Figure_2.png?raw=true "Model performance 2")
![Figure 3 displaying a comparison between the down-scaled image, up-scaled image by the model, and the original image.](images/Figure_3.png?raw=true "Model performance 3")
![Figure 4 displaying a comparison between the down-scaled image, up-scaled image by the model, and the original image.](images/Figure_4.png?raw=true "Model performance 4")
![Figure 5 displaying a comparison between the down-scaled image, up-scaled image by the model, and the original image.](images/Figure_5.png?raw=true "Model performance 5")