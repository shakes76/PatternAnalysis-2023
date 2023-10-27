## Project Summary
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
##sylegan
StyleGAN, short for Style Generative Adversarial Network, is a state-of-the-art generative model used for creating high-quality synthetic images. It was first introduced by the engineers at NVIDIA in 2008. It revolutionized the field of generative adversarial networks (GANs) with its ability to generate highly realistic and diverse images. The main characteristic of StyleGAN lies in its ability to control the style and appearance of generated images, offering fine-grained manipulation of visual elements such as facial features, background scenery, and more. It does so by introducing a concept of "style," using a progressive growing approach, and utilizing a high-dimensional latent space to produce images that can seamlessly interpolate between various visual attributes. Additionally, StyleGAN incorporates noise inputs, a discriminator network to assess image authenticity, and a unique architecture that progressively increases image resolution for better results.

##styleGAN2

StyleGAN2 introduced several improvements, including a two-part generator architecture consisting of a mapping network and a synthesis network, which allows for better control over the style of generated images. It also introduced the concept of "minibatch standard deviation" to encourage image diversity during training, leading to more realistic and varied outputs. Another significant enhancement was the equalized learning rate, which provided better training stability and balanced contributions from different layers of the network. StyleGAN2 offered improved performance, higher image quality, and greater control over the generated content, making it a go-to choice for a wide range of applications, including art, image manipulation, and data augmentation. These advancements established StyleGAN2 as a cutting-edge solution for high-quality image synthesis.

## Model Architecture






