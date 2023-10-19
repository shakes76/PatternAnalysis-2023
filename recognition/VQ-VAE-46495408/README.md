# Generative Model of OASIS Brain Data Using VQ-VAE & PixelCNN
## Overview
This project constructs a generative model that can generate OASIS brain images using the VQ-VAE (Vector-Quantized Variational Autoencoders) model and PixelCNN model. First, use VQ-VAE to learn the OASIS dataset for optimizing encoding embeddings. Then, use the trained encoder to generate a codebook for training the PixelCNN model. Finally, randomly generate a new encoding matrix through PixelCNN, map it through the codebook, and decode it to obtain a new image.
## VQ-VAE Model
<image src="" width="">
Unlike variational encoders and autoencoders, the latent space in VQ-VAE is discrete rather than continuous, which can prevent posterior collapse. Additionally, the prior probability distribution in VQ-VAE is not static and can be learned through training. The encoder uses a convolutional network to generate corresponding features, then calculates the Euclidean distance, and maps the vectors to a discrete codebook. The decoder retrieves the corresponding code from the codebook based on the most recent embedding and uses the code words for data generation. 
  
### Gradient design
<image src="" width="">
The backpropagation of VAE is challenging because the process of obtaining the nearest code word during the forward pass is non-differentiable. The paper uses a straight-through estimator because the encoder and decoder have the same dimensions. By copying gradients and adjusting the direction of the corresponding encoding vectors, the encoder's output is continuously moved closer to the nearest code.
  
### Loss function
<image src="" width="">

The loss function consists of three parts. 
1. Reconstruction loss: The difference between the encoder's input and the generator's output.
2. Loss for optimizing the encoding embedding: The code continuously moves closer to the input, learning the embedding.
3. Loss for optimizing the encoder's output, approaching the code in the codebook.

### Architecture
Due to the use of full-sized grayscale images, the input shape of the encoder is (n, 256, 256, 1). For the vector quantizer layer, I use 32 embeddings with a dimension of 128.
<image src="" width="">

### Results
<image src="" width="">
<image src="" width="">
<image src="" width="">

## Pixel CNN Model
### Architecture

## Result

## Dependencies

## References
*
*
*
