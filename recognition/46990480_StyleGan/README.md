# COM3710 Stable Diffusion Report
An implementation of StyleGAN2 for generating images of the human brain based on the [OASIS brains](https://www.oasis-brains.org/) dataset. 

## The Problem
The sophistication and complexity of the human brain has fascinated scientists for hundreds
of years. With the recent momentum surrounding generative AI, it is time to harness this
technology and produce images of brains. The objective of this project was to design a
StyleGAN2 implementation with the purpose of generating *reasonably clear* 256x256px images 
of the human brain. 

## Why a GAN?
A Generative Adversarial Network (or GAN) is an deep learning model comprised of two 
convolutional networks, a Generator network and a Discriminator network. The Generator 
network attempts to generate 'fake' images from random noise input in order to trick the 
discriminator network into thinking that the produced image is 'real'. Mathematically 
speaking, the two networks are playing a min-max game whereby we are trying to maximise the 
loss of the discriminator and minimise the loss of the generator.

## Dataset

## Requirements

### Code Structure

## Model Implementation

## Training & Results

## Where to Next?

## References & Acknowledgements
- https://www.oasis-brains.org/