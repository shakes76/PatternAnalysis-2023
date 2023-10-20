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
      our codebook takes its sweet time. 
    - **Codebook Loss**: This is simply the L2-norm error, which nudges our embedding or codebook vectors to align better with the encoder's output.
3. Reconstruction Loss: At the end of the day, we want our reconstructed image to resemble the original. This loss measures how well we're doing in that aspect.

   The formula for the total loss can be represented as:

**Total Loss** = Reconstruction Loss + VQ Loss

Where:

**VQ Loss** = Commitment Loss + Codebook Loss


