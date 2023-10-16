# StyleGAN model on the OASIS brain dataset

Author: Zhixuan Li, s4614899

## Task
The task is to utilize a generative model on the OASIS brain dataset, with the goal of generating "reasonable clear images".

## Model
A styleGAN model is constructed and used for the given task, which is a state-or-art variation of the Generative Adversarial Network(GAN). Built upon regular GAN models, styleGAN is designed to further improve the quality as well as the control of generated images by introducing several extra components:
- Style Mapping:
  Instead of directly feeding the latent space vector *z* into the generator, which might cause the issue of feature entanglement, styleGAN firstly converts the *z* to an intermidiated latent space *w* (also known as the style factor) via a mapping network, in order to untangle the data distribution so that training the generator could be eaiser.
- Adaptive Instance Normalization:
  AdaIN is essentially a normalizaton technique that aligns the mean and variance of the content feature with that of the style feature in the generator of styleGAN model. It helps to modulate and manipulate generated images based on the style factor *w*.
- Progressive Training:
  The training of styleGAN starts with low-resolution images (4*4 in this case) and progressively increases the resolution by adding new layers until it reaches the resolution of the original images to be resembled (256*256). This approach accelerates and stablizes the training process.
- Stochastic Variation as Noise input:
  Stochastic variation is introduced to different layers of the generator using scale/resolution-specific Gaussian noises, where the scale-specificity is achieved by the learnable scaling factor (represented as the weight variable in the code), allowing for fine details such as hairs and freckles to be generated.

## Input images
The Open Access Series of Imaging Studies (OASIS) contains thousands of MRI image data of brain, which are used as input data for the styleGAN model, here is a sample of the OASIS data output by the dataset.py file:
![Sample images from OASIS dataset: ](./output_images/sample_grid.png)

## Pre-processing
Rather than directly importing the dataset, the dataset.py file specifically handles the OASIS brain data, which is stored on rangpur as three seperate data files for training, testing and validation purposes, respectively, by reading all the images with a few transformations applied.

## Model training

## Generated images
After training the styleGAN model, here is a sample of generated images output by the predict.py file:
![Generated images by style generator: ](./output_images/generated_grid.png)

## Issues


## Solution


## Dependencies

## Reference
- Original paper on styleGAN: https://doi.org/10.48550/arXiv.1812.04948
- Referenced implementation of styleGAN model: https://www.kaggle.com/code/tauilabdelilah/   stylegan-implementation-from-scratch-pytorch
