Stable Diffusion on OASIS Dataset
===

> The readme file should contain a title, a description of the algorithm and the problem that it solves(approximately a paragraph), how it works in a paragraph and a figure/visualisation.

# Requirements (This should be removed when submitted)

1. The readme file should contain a title, a description of the algorithm and the problem that it solves(approximately a paragraph), how it works in a paragraph and a figure/visualisation.
2. It should also list any dependencies required, including versions and address reproduciblility of results,if applicable.3.
3. provide example inputs, outputs and plots of your algorithm
4. The read me file should be properly formatted using GitHub markdown
5. Describe any specific pre-processing you have used with references if any. Justify your training, validationand testing splits of the data.

# Project Overview

## Results

## Diffusion Process GIF


# About Stable Diffusion 

## What is Diffusion Model?
> [DDPM (Denoising Diffusion Probabilistic Models)](https://arxiv.org/abs/2006.11239)


## What is Stable Diffusion?
> [High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)](https://arxiv.org/abs/2112.10752)

![](report_imgs/stable_diffusion_flowchart.png)

# Detail in Stable Diffusion


## Perceptual Image Compression

#### Swish
> [SEARCHING FOR ACTIVATION FUNCTIONS, ICLR 2018 workshop](https://arxiv.org/pdf/1710.05941.pdf)

Swish is an activation, which is defined as $f(x) = x \cdot \sigma ( \beta x)$.

In our code, we set $\beta = 1$ in all the Swish activation, and which is also known as "Sigmoid Linear Unit (SiLU)".

* Update: Pytorch has implementation of [SiLU](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html)

![](report_imgs/swish.png)
> The swish function 

#### Group Normalization

#### Linear Attention

#### ResNet

#### Reparameterization trick on VAE


## Latent Diffusion Models

## Conditioning Mechanisms

#### Time Embedding