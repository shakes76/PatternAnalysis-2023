# COMP3710 - Pattern Analysis Report

**Student Number:** s4698512

**Chosen project:** #9 - Vector Quantised Variational Auto Encoder (VQ-VAE) on both OASIS and ADNI Datasets.

**Due Date:** Monday, $16^{\text{th}}$ October, 2023 (Week 12)

## Description of the VQ-VAE Model

The VQ-VAE (or Vector Quantised Variational Auto-Encoder) is a modified version of the well known auto-encoder. In essence, an auto-encoder encodes or compresses an input image down to a latent space, before decoding this in an attempt to recover as much of the input image as possible. A good auto-encoder will minimise the reconstruction loss, leading to applications including but not limited to

-   Image Compression
-   Denoising
-   Image Generation

A visualisation of the VQ-VAE model structure is given in the following diagram.

![VQ-VAE Model Structure](Assets/vqvae_model_structure_cropped.png)

_Figure 1: VQ-VAE model structure. Source: [arXiv [2]](https://arxiv.org/abs/1711.00937v2)_

The VQ-VAE model differs in three main ways from a standard auto-encoder. Firstly, instead of mapping inputs to discrete vectors within the latent space, it instead learns a Gaussian normal distribution. As a result, the latent space becomes, for one, continuous and differentiable which is useful for backpropagation and gradient-based optimisation, but more importantly, far more interpolatable - a highly useful property for generative models.

Secondly, the vector-quantised part of the VQ-VAE discretises the latent space into $K$ discrete vectors and stores these in a codebook. This seems counterintuitive as the main purpose of a VAE is to make the latent space continuous, however, the learnt discrete latent space distriution of a VQ-VAE, is still regularised, yet it has the additional benefit of learning discrete features of the data it may be learning.

Finally, a reparameterisation trick, similar to parametric bootstrap, is used during model training to overcome the now discrete latent space being non-differentiable. Instead of sampling from the codebook directly, the codebook is modelled as a normal distribution meaning that the latent space can be differentiable during training for the purpose of backpropagation and gradient descent. The discrete codes, however, are still used to represent the data which provides the advantages outlined above.

---

## It should also list any dependencies required, including versions and address reproduciblility of results, if applicable.

## provide example inputs, outputs and plots of your algorithm

## Preprocessing

Describe any specific pre-processing you have used with references if any. Justify your training, validation and testing splits of the data.

# References

1. Neural Discrete Representation Learning
   Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu
   https://doi.org/10.48550/arXiv.1711.00937

2. https://arxiv.org/abs/1711.00937v2

3. https://github.com/ritheshkumar95/pytorch-vqvae/
