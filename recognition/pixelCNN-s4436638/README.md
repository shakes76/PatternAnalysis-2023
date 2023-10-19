#Brain MRI Super-Resolution Network
## Abstract

This project aimed to create an efficient sub-pixel neural network (or ESPCN) as proposed by Shi, 2016, to upscale low resolution images into a higher resolution version. It will use sub-pixel convolution layers to create an array of upscaling filters, which can then be used increase the resolution of our images. In the case of our project, this will be applied to brain MRI scan images - in particular, the ADNI brain dataset will be used to train our CNN. We will take existing images from the dataset, downsample them by 4x, and use these to train our network to upscale them back to the original resolution. 

## Model Architecture

The used model utilises a combination of convolution layers for feature maps extraction, and sub-pixel convolution layer to collate these maps and upscale our image. Specifically, this ESPCN will use multiple convolution layers, uses the ReLU function for activation, and the PixelShuffle function to aggregate our channels. The diagram below illustrates this model in action.

![image](https://github.com/CharlieGore/PatternAnalysis-2023/assets/141538622/b79ce09f-9464-4734-8f29-090b08ec5295)

According to the diagram, the low-resolution image undergoes a process of generating multiple feature maps, which are subsequently combined into a single channel to produce the high-resolution image. The number of filters created is determined by squaring the upscaling factor, meaning that our project, which upscales by a factor of 4, will result in the creation of 16 filters. Previous research has demonstrated that the Adam optimizer consistently delivers the most favorable outcomes during model training, and the mean square error loss function has proven to be the most effective method for assessing the system's loss.

## Dataset and preprocessing

The dataset utilised was the ADNI brain dataset, consisting of brain MRI images sized at 240x256 pixels. As we want to use different images for training and validation, this dataset was split into 2 arrays, with 90% of the image dataset being used for training and the other 10% for validation. These images also had to be downsampled using the Resize function by our upsampling factor of 4, making them 60x64 when being used in the CNN.

## Training

The trainer loads in the original ground truth images in batches of 10, at 240x256, to avoid storing the whole dataset in RAM. An example ground truth image:

![image](https://github.com/CharlieGore/PatternAnalysis-2023/assets/141538622/e906d64c-de25-4f03-82ff-7f629b931fa6)



![image](https://github.com/CharlieGore/PatternAnalysis-2023/assets/141538622/5aea520e-f313-4a36-ba7d-471ba140d947)



![image](https://github.com/CharlieGore/PatternAnalysis-2023/assets/141538622/26954310-bd30-4172-a936-47e245f2a281)





