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
