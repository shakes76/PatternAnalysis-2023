# VQ-VAE with PixelCNN for Brain Image Reconstruction and Generation

## Introduction

This project explores the use of Vector Quantized Variational Autoencoders (VQ-VAE) combined with PixelCNN for the purpose of brain image reconstruction and generation. The code demonstrates the complete pipeline, starting from dataset loading and preprocessing, to model training, and finally visualization of results.

## Dataset

The dataset consists of brain slice images. The data is zipped and stored in Google Drive, but can be easily extracted and processed for use in the project. 

The dataset is split into:
- Training: 9664 images
- Testing: 544 images
- Validation: 1120 images

Each image is of shape 128x128.

![Dataset Samples](path_to_dataset_samples_image)

## Model Definitions

### VQ-VAE

VQ-VAE is used for the compression of brain images. It comprises three main components: an encoder, a vector quantizer, and a decoder. The encoder maps input images to a continuous representation, which is then quantized by the vector quantizer. The quantized representation is finally mapped back to the original image space using the decoder.

When we talk about the loss in the VQ-VAE model, it's a blend of three primary components:

1. **Total Loss**: This is like the grand total on a bill. It combines the losses from the vector-quantization layer and the image reconstructions.
2. **Vector Quantization (VQ) Loss**: This is further split into two parts:
    - **Commitment Loss**: This ensures that the encoder remains loyal to a particular codebook. It's essential because while our encoder learns pretty quickly,
      our codebook takes its sweet time. The commitment loss is like a gentle nudge to ensure they remain in sync. We also introduce a scaling factor, termed as the beta parameter.
      Even though the original VQ-VAE paper mentioned that the model is sturdy against changes in this parameter, it still plays a role in the commitment.
    - **Codebook Loss**: This is simply the L2-norm error, which nudges our embedding or codebook vectors to align better with the encoder's output.
3. **Reconstruction Loss**: At the end of the day, we want our reconstructed image to resemble the original. This loss measures how well we're doing in that aspect.

   The formula for the total loss can be represented as:

**Total Loss** = Reconstruction Loss + VQ Loss

Where:

**VQ Loss** = Commitment Loss + Codebook Loss

### PixelCNN
PixelCNN is like an artist with a paintbrush, creating images one pixel at a time. It's a generative model that cleverly utilizes convolutional and residual blocks. 
The idea is to compute the distribution of prior pixels to guess the next pixel.

**How it Works:**
1. **Initial Convolution**: The input image is passed through a convolutional layer.
   This process is a bit like using a magnifying glass to inspect the image, where the "receptive fields" help the model learn features for all the pixels simultaneously.
   But there's a catch! We use masks, termed 'A' and 'B', to ensure that we're not "cheating" by looking at pixels we shouldn't.
   The 'A' mask restricts connections to only the pixels we've already predicted, while the 'B' mask allows connections only from predicted pixels to the current ones.
3. **Residual Blocks**: After the initial convolution, the data flows through residual blocks.
   These blocks are smart! Instead of trying to learn the output directly, they focus on learning the difference (or residuals) between the expected output and the current one.
   This is achieved by creating shortcuts (or skip connections) between layers.

### The Loss Mechanism:
For PixelCNN, the loss metric used is the Sparse Categorical Crossentropy loss. This quantifies the error in selecting the right latent vectors (or pages from our codebook) for image generation.
PixelCNN is a generative model trained to predict the next pixel's value in an image given all the previous pixels. It's employed post-VQ-VAE training to refine the generated images, making them more realistic.




