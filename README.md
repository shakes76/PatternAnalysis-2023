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

Finally, in stable diffusion, this paper introduces a skip connection between the input and its self-attention's output, which means that the output of self-attention will be $X + \text{proj}(\text{ATTN}(X))$, where "proj" is a one-by-one convolutional layer.

![](https://miro.medium.com/v2/resize:fit:1400/1*1je5TwhVAwwnIeDFvww3ew.gif)
> Flow chart of Self-Attention




#### Reparameterization trick on VAE

#### LPIPS
#### VAE GAN
#### PatchGAN

####  Overview

* Construct an Unet-like autoencoder, and it used some module like Swish, Self-attention and resnetblock
* Use reparameterization in latent space to calculate KL divergence.
* Use GAN (or discriminator) with LPIPS score to optimize decoder.



## Latent Diffusion Models

## Conditioning Mechanisms

#### Time Embedding