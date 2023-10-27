# Project Summary
This project utitlizes StyleGAN2 to produce fake images that are almost indistinguishable from real images. The dataset for the model is portion of the OASIS brain Dataset which includes T1-weighted MRI scans obtained in single scan sessions.

## Purpose
StyleGAN2's application in brain image generation is driven by the need for realistic and versatile synthetic brain scans in medical and research contexts. Its primary purpose is to provide a powerful tool for creating diverse and customizable brain images. By offering fine-grained control and reducing privacy concerns, StyleGAN2 accelerates research, aids in the development of diagnostic tools, and addresses data scarcity issues, ultimately benefiting the field of neuroscience and medical imaging.

## Overview
The folder comprises of 5 files.
config.py: Comprises of all the hyperparameters and predefined settings for the model training.
dataset.py: Reads data, augments and load the datasets to tensors.
modules.py: Implements the Generator and the Discriminator. And their subsequent layers.
predict.py: Generates the fake image from the learning / leant generator

#model

## SyleGAN
StyleGAN, short for Style Generative Adversarial Network, is a state-of-the-art generative model used for creating high-quality synthetic images. It was first introduced by the engineers at NVIDIA in 2008. It revolutionized the field of generative adversarial networks (GANs) with its ability to generate highly realistic and diverse images. The main characteristic of StyleGAN lies in its ability to control the style and appearance of generated images, offering fine-grained manipulation of visual elements such as facial features, background scenery, and more. It does so by introducing a concept of "style," using a progressive growing approach, and utilizing a high-dimensional latent space to produce images that can seamlessly interpolate between various visual attributes. Additionally, StyleGAN incorporates noise inputs, a discriminator network to assess image authenticity, and a unique architecture that progressively increases image resolution for better results.

## StyleGAN2

StyleGAN2 introduced several improvements, including a two-part generator architecture consisting of a mapping network and a synthesis network, which allows for better control over the style of generated images. It also introduced the concept of "minibatch standard deviation" to encourage image diversity during training, leading to more realistic and varied outputs. Another significant enhancement was the equalized learning rate, which provided better training stability and balanced contributions from different layers of the network. StyleGAN2 offered improved performance, higher image quality, and greater control over the generated content, making it a go-to choice for a wide range of applications, including art, image manipulation, and data augmentation. These advancements established StyleGAN2 as a cutting-edge solution for high-quality image synthesis.

## Model Architecture

**Generator Network:** The generator network is responsible for creating synthetic images. It takes a random noise vector as input and produces an image as output. StyleGAN uses a progressive growing approach, where the generator consists of multiple layers that increase in resolution gradually. Each layer uses a convolutional neural network (CNN) architecture to transform the input noise into an image.


**Discriminator Network:** The discriminator network is used to distinguish between real and generated images. It is also a CNN, and its role is to assess the authenticity of the images produced by the generator. The discriminator is trained to minimize the distinguishability of generated images from real ones.


**Latent Space:** StyleGAN operates in a high-dimensional latent space, where each point in the space corresponds to a unique image. The generator network maps points in this latent space to images, allowing for smooth interpolation between different images by interpolating between points in the latent space.


**Style Mapping Network:** StyleGAN introduces the concept of "style" into the generative process. It uses a separate style mapping network to map the latent code to a set of style vectors. These style vectors control various aspects of the generated image, such as the features and appearance.


**Noise Inputs:** StyleGAN also incorporates noise inputs at various layers of the generator network. This noise adds stochastic variations to the images and contributes to their diversity and realism.


**Progressive Growing:** StyleGAN uses a progressive growing technique, where the generator and discriminator networks start with low-resolution images and gradually increase the resolution as training progresses. This approach helps in generating high-quality images with fine details.
Mapping Network and Synthesis Network: StyleGANv2 introduced a two-part generator architecture, consisting of a mapping network that converts the latent code into style vectors and a synthesis network that generates the image. This separation of concerns allows for better control over the style of generated images.


**Equalized Learning Rate:** StyleGAN often employs an equalized learning rate for different layers of the network. This helps in training stability and ensures that the contributions of different layers are balanced.


**Minibatch Standard Deviation:** To encourage diversity in generated images, StyleGANv2 introduced the concept of minibatch standard deviation, which measures the diversity of feature statistics across a minibatch of images during training.
These components work together to create a generative model capable of producing high-quality, diverse, and realistic synthetic images. StyleGAN has been widely used in various applications, including art generation, image editing, and data augmentation.






