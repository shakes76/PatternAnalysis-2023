# Generative Model of OASIS Brain Data Using VQ-VAE & PixelCNN
## Overview
This project constructs a generative model that can generate OASIS brain images using the VQ-VAE model and PixelCNN model.
First, use VQ-VAE to learn the OASIS dataset for optimizing encoding embeddings. 
Then, use the trained encoder to generate a codebook for training the PixelCNN model. 
Finally, randomly generate a new encoding matrix through PixelCNN, map it through the codebook, and decode it to obtain a new image
## VQ-VAE Model

### Architecture

### Results

## Pixel CNN Model
### Architecture

## Result

## Dependencies

## References
*
*
*
