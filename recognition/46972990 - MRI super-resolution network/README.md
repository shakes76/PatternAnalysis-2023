# Implementation of a Brain MRI Super-Resolution Network (ESPCN)

## Introduction
In this recognition task, a super-resolution network model will be created that can up-scale by a factor of 4 on downsampled ADNI brain MRI scan images to produce a reasonably clear image. The dataset used for this task will be the [ADNI brain dataset](https://adni.loni.usc.edu/). The algorithm used to complete this recognition task is the [ESPCN (Efficient Sub-Pixel CNN)](https://keras.io/examples/vision/super_resolution_sub_pixel/). This algorithm reconstructs a high-resolution version of an image by leveraging
efficient sub-pixel convolution layers to learn image upscaling filters. In the case of this recognition task, the ESPCN model will take down-scaled images (by a factor of 4) from the ADNI brain dataset and attempt to up-scale the images back to their original resolution without any perceptual loss of quality. 

![Figure 1 displaying a comparison between the down-scaled image, up-scaled image by the model, and the original image.](images/Figure_1.png?raw=true "Model performance 1")
![Figure 2 displaying a comparison between the down-scaled image, up-scaled image by the model, and the original image.](images/Figure_2.png?raw=true "Model performance 2")
![Figure 3 displaying a comparison between the down-scaled image, up-scaled image by the model, and the original image.](images/Figure_3.png?raw=true "Model performance 3")
![Figure 4 displaying a comparison between the down-scaled image, up-scaled image by the model, and the original image.](images/Figure_4.png?raw=true "Model performance 4")
![Figure 5 displaying a comparison between the down-scaled image, up-scaled image by the model, and the original image.](images/Figure_5.png?raw=true "Model performance 5")

## Model Architecture
![Model architecture](images/Digraph.png?raw=true "Torchviz visualisation of the ESPCN model.")

## Model Description
The ESPCN model is specifically designed for image super-resolution tasks. It uses convolutional layers to extract hierarchical features from the low-resolution input image, then it uses a final convolutional layer followed by a pixel shuffling to up-scale the image to the desired resolution (back to the original image's resolution for this specific task). The use of the pixel shuffle operation makes this method efficient and allows it to achieve good super-resolution performance with relatively few parameters. The model itself consists of:

* Conv1: A convolutional layer that takes a grayscale image (channels = 1) and outputs 64 feature maps whilst using a 5x5 kernel size with padding of 2 and using reflection padding mode. The reflection padding mode is used to reduce border artifacts in the case that a brain MRI image is touching the border of the image.
* Conv2: The second convolutional layer that takes the 64 feature maps from before and outputs another 64 feature maps. It uses a 3x3 kernel with padding of 1.
* Conv3: The third convolutional layer that takes the 64 feature maps and outputs 32 feature maps using a 3x3 kernel and with padding of 1.
* Conv4: This fourth convolutional layer takes the 32 feature maps and produces (channels * (upscale_factor ** 2)) feature maps. In this case of this particular recognition task, the channels is 1 (since grayscale) and the upscale factor is 4. So the output of this convolutional layer is 16 feature maps.
* PixelShuffle: This layer rearranges elements in the feature map from the depth dimension to the spatial dimensions, thereby achieving upscaling.
* RELU: The Rectified Linear Unit activation function introduces non-linearity after each convolutional layer.