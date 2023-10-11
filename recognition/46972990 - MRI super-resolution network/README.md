# Implementation of a brain MRI super-resolution network
Benjamin Guy 46972990

## Introduction
In this recognition task, a super-resolution network model will be created that can up-scale by a factor of 4 on downsampled ADNI brain MRI scan images to produce a reasonably clear image. The dataset used for this task will be the [ADNI brain dataset](https://adni.loni.usc.edu/). Down-sampled data with a factor of 4 will be created from the data for training, with the model expected to up-scale this data back to its original resolution.

## Research
There is a number of existing solutions used to up-scale images. Some examples include the [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](
https://doi.org/10.48550/arXiv.1809.00219) and the [Image Super-Resolution using an Efficient Sub-Pixel CNN](https://keras.io/examples/vision/super_resolution_sub_pixel/). The current implementation of the ESRGAN is quite extensive and requires knowledge on generative adversarial networks, which may prove to be overkill for the given task at hand. As such, the Sub-Pixel CNN method will be used initially and if results prove to be unsatisfactory, the model will shift towards an implementation of the ESRGAN model.