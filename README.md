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

## Run

## Files

### Non-Script
* `model_ckpt\*`: models' checkpoints.
* `log\*`: training log. (Can use `logger.py` to visualize it.)
* `*_vis\*`: visualization of images when training.
* `playground`: we write some code to figure out some algorithms.
* `README.md / report.pdf`: The file you current see.

### Common
* `util.py`: useful function, such as ssim, positional encoding.
* `module.py`: useful network module, such as resblock.
* `dataset.py`: get dataset from given folder.
* `logger.py`: script that can check the training log.

### First Stage
* `model_VAE.py`: `VAE` and `VQVAE` model for first stage.
* `model_discriminator.py`: The GAN part when training `VAE` or `VQVAE`.
* `prestage_train.py`: Training Script for first stage.

### Second Stage
* `pixelCNN.py`: Do pixelCNN when random generation from VQVAE model.
* `model_diffusion.py`: The model of stable diffusion. It's UNet.
* `stable_diffusion.py`: Do stable diffusion when random generation from `VQVAE` or `VAE` model.

##### Second Stage Script 


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

![Swish Function](report_imgs/swish.png)

#### Group Normalization
> [Group Normalization, ECCV 2018](https://arxiv.org/abs/1803.08494)

![](report_imgs/group_normalization.png)

Normalization: A method to train model faster and more stable through normalization of tinputs by re-centering and re-scaling.
* Batch normalization: Normalization for each channel.
* Layer normalization: Normalization for each sample.
* Instance normalization: Normalization for each sample and each channel.
* Group normalization: Normalization for each sample group.

> Note: If batch size is large enough, the performance: BN > GN > LN > IN 
> However, BN has GPU memory issue and cannot set large batch size sometimes. 
> Thus, we do GN in this task.

#### ResNet
> [Deep Residual Learning for Image Recognition, CVPR 2016](https://arxiv.org/pdf/1512.03385.pdf)

![](report_imgs/residual_block.png)

A popular module to very learn deep model by residual learning.

#### Self-Attention
> [Attention Is All You Need, NIPS 2017](https://arxiv.org/abs/1706.03762)

Self-attention, also known as scaled dot-product attention, is a crucial concept in the field of natural language processing (NLP) and deep learning, particularly within the context of transformer-based models. In stable diffusion, authors employ self-attention in each feature map. Next, we will introduce what self-attention is.

Self-attention involves three key vectors for executing "attention to itself":

1. Query vectors: These query other elements in the sequence.
2. Key vectors: These define the importance relative to the current element.
3. Value vectors: These generate the output vectors.

After generating these three vectors, we follow these steps for each element:

1. Calculate the attention score, which is $W_{current, i} = K_{current} \cdot Q_{i}$, where $i \in [1, C]$, and $C$ is the input size.
2. Scale and softmax the score, which means $W_{current} = \text{softmax}\left(\frac{W_{current}}{\sqrt C}\right)$.
3. Generate the output, $H_{current} = W_{current} * V$.

Finally, to apply this method to convolutions, we can simply replace all linear layers with one-by-one convolutional layers, and everything will be fine.

![](https://miro.medium.com/v2/resize:fit:1400/1*1je5TwhVAwwnIeDFvww3ew.gif)
> Flow chart of Self-Attention

#### Variational Autoencoder (VAE)
> [Auto-Encoding Variational Baye, ICLR 2014](https://arxiv.org/abs/1312.6114)

This paper introduce what VAE is.

![](https://miro.medium.com/v2/resize:fit:1200/1*Mp7gBL3qMLyWZDiNEC_EpQ.png)
> Flow chart of VAE.

The most important thing that VAE aims to solve is to make it easier for us to generate random images from the latent space. We need to know what distribution the latent space, encoded by the encoder, follows, but this problem is quite challenging. Therefore, we can approach it from a different perspective: constraining the latent code generated by the encoder to be similar to a well-known distribution, typically a Gaussian distribution.

But how do we calculate the regularization term? Next, we will explore what the regularization term should be if we use the KL divergence to measure the distance between the latent space and a Gaussian distribution.

Notation: $q$ is encoder, $z$ is latent.

* For an showed image $x$, we need to maximize the log probability (because of max likelihood) $x$ generated by AE, $p(x)$.

  * $\log p(x) = \log \int_z p(x, z) dz = \log \int_z p(x, z) \frac{q(z|x)}{q(z|x)} dz$
  * By Jensen's inequality, $\log p(x) \ge \mathbb E_{q(z|x)} \log \frac{p(x, z)}{q(z|x)} dz$
  * By Bayes' theorem, $\log p(x) \ge \mathbb E_{q(z|x)} \log \frac{p(x|z) p(z)}{q(z|x)}$
    * $= \mathbb E_{q(z|x)} \log p(x|z)  + \mathbb E_{q(z|x)} \log \frac{p(z)}{q(z|x)}$
    * $= \mathbb E_{q(z|x)} \log p(x|z)  + \int_z q(z|x) \log \frac{p(z)}{q(z|x)}$
    * $= \mathbb E_{q(z|x)} \log p(x|z)  - D_{KL}[q(z|x)||p(z)]$
    * $= Maxlikelihood - D_{KL}[q(z|x)||p(z)]$

* Where we assume $p(z)$ is gaussian distribution $N(0, 1)$, and $q(z|x)$ is $N(\mu, \sigma)$ 

  * Recall that $N(\mu, \sigma) = \frac{1}{\sigma \cdot \sqrt {2\pi}} \exp(-\frac  12 (\frac {x-\mu}{\sigma})^2) $

    * $\log N(\mu, \sigma) = \log \frac{1}{\sqrt{2\pi}} - \log \sigma -\frac  12 (\frac {x-\mu}{\sigma})^2 $

  * $D_{KL}[N(\mu, \sigma)||N(0, 1)] = \int N(\mu, \sigma) [ (\log \frac{1}{\sqrt{2\pi}} - \log \frac{1}{\sqrt{2\pi}}) - (\log \sigma - \log 1)  - \frac{1}{2}((\frac{x - \mu}{\sigma})^2 -(\frac{x - 0}{1})^2)] dx$

    * $= \mathbb E_{N(\mu, \sigma)} [ - \log \sigma  - \frac{1}{2}((\frac{x - \mu}{\sigma})^2 - x^2)]$
    * $= -\log \sigma -\frac {1}{2\sigma ^2} \mathbb E_{N(\mu, \sigma)}[(x - \mu)^2] +  \mathbb  E_{N(\mu, \sigma)}[x^2]$
    * $= -\log \sigma - \frac 12 +  \mathbb  E_{N(\mu, \sigma)}[x^2]$
      * $\sigma^2 = \mathbb E[x^2] - \mathbb  E[x]^2 \rightarrow \mathbb E[x^2] = \mu^2 + \sigma^2$
    * $= - \log \sigma - \frac 12 - \frac 12( \sigma^2 + \mu^2)$
    * $= -\frac{1}{2} (1 + \log  \sigma^2 -\sigma^2 - \mu^2)$
  * Thus, the regularization term for KL divergence is $= -\frac{1}{2} (1 + \log \sigma^2 - \sigma^2 - \mu^2)$. If we modify the encoder from an image to some $\mu$ and $\sigma$, and sample the latent variable from $N(\mu, \sigma)$, then we can apply this term.

#### KLD Loss Scheduler

> [Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing (NAACL 2019)](https://github.com/haofuml/cyclical_annealing)

Balancing the weights of VAE_loss and KLD_loss is a challenging task. If we set VAE_loss too high, clear images won't be generated from sampling the latent space because it won't match the decoder's distribution. Conversely, if we set KLD_loss too high, the autoencoder may tend to experience mode collapse and ignore the reconstruction loss term. To mitigate this issue, we employ a technique known as the cyclical annealing scheduler, which periodically adjusts the weight of KLD_loss.

![](https://github.com/haofuml/cyclical_annealing/raw/master/figs/NAACL_2019_Cyclical.png)

#### VQVAE

#### VQGAN

#### PatchGAN

####  Overview
* Construct an Unet-like autoencoder, and it used some module like Swish, Self-attention and resnetblock
* Use reparameterization in latent space to calculate KL divergence.
* Use GAN (or discriminator) with LPIPS score to optimize decoder.
  * LPIPS score is adopted in stable diffusion. But we have no pre-trained model to use.



## Latent Diffusion Models

## Conditioning Mechanisms

#### Time Embedding