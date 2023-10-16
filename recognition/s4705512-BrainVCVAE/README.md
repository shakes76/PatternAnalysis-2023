
# OASISVQ: Enhancing Brain Image Generation with VQVAE

OASISVQ is a project dedicated to advancing brain image generation using the Vector Quantized Variational Autoencoder (VQVAE) deep learning model. This README provides an overview of the project and its goals.

# Ploblem Overview
Brain image generation is a crucial task in neuroimaging research, aiding in the study of brain structure, function, and various neurological conditions. OASISVQ aims to utilize the capabilities of VQVAE, a powerful variant of Variational Autoencoders, to enhance the generation of brain images from the OASIS Brain dataset.

# Objectives
Implement a VQVAE model for brain image generation.
Achieve a "reasonably clear image" with a Structured Similarity (SSIM) score exceeding 0.6.
Provide a valuable tool for researchers to generate realistic brain images for diverse applications in neuroimaging studies.

# Dataset
Data were provided by OASIS in [this resouce](https://www.oasis-brains.org/#data)
The OASIS Brain dataset is used as the foundation for training and evaluating the VQVAE model. This dataset encompasses structural and functional MRI scans, covering a wide range of subjects, including healthy individuals and those with neurological conditions.

# Model Architecture
Encoder:
Utilizes a convolutional neural network (CNN) to transform input brain images into a lower-dimensional latent space.
Vector Quantization Layer:
Maps continuous latent representations to discrete vectors using a codebook, enhancing the diversity and quality of generated images.
Embedding Layer:
Transforms the quantized latent representation for further processing.
Decoder:
Reconstructs the original brain image from the quantized latent representation, producing realistic and diverse outputs.

# Requirements and Dependency
matplotlib >= 3.5.2
numpy >= 1.21.5
requests >= 2.28.1
tensorflow >= 2.10.1
tensorflow-probability >= 0.14.0
python >= 3.7.13



## References
- https://en.wikipedia.org/wiki/Structural_similarity
- https://keras.io/examples/generative/vq_vae/
- https://keras.io/examples/generative/pixelcnn/
- https://arxiv.org/abs/2101.08052

- OASIS-1: Cross-Sectional: Principal Investigators: D. Marcus, R, Buckner, J, Csernansky J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382
- OASIS-2: Longitudinal: Principal Investigators: D. Marcus, R, Buckner, J. Csernansky, J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382
---
# other 
