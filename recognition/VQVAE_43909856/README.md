# Insert title here

This file will contain the report contents.

## Description
Include a description of the algorithm and the problem that it solves (a paragraph or so).

Include how the model works (in a paragraph or so).

Include a figure/visualisation of the model.

## Dependencies
List any required dependencies, including versions. 

Address reproducibility of results (if applicable).

## Examples
Provide example inputs and outputs. 

Provide plots of the algorithm.

## Preprocessing
Describe any specific preprocessing used (if any) with references. 

Justify the training, validation, and testing splits of the data.



### Notes
VQVAEs:
- Mathematically grounded (MSE loss)
- Aiming to simplify distribution P(x) (the likelihood to be maximised) using a Variational Autoencoder
- Reconstruction: decoder learns P(x|z)
- Prior learns P(z) auto-regressively
- Combined: P(x)
- Auto-regressive sampling can be very slow (32x32 requires 1024 steps)
- 1D vector quantisation: Efficiently encode signal vectors (compress them). Only need to transmit indices (assuming both sides have the codebook/dictionary to decode). Inherently lossy
- n-D vector quantisation: Use for image compression (JPEG) - 3D RGB signal. JPEGs only retains a 'dictionary' of limited colours. More nuanced colours get mapped to the closest colour in the 'dictionary'. Lossy compression can result in artefacts
- Learn a dictionary of latent vectors (codebook) - vectors are trainable weights in a NN
- Map encoder output to the entries in the codebook by Euclidean distance
- Decode dictionary entries to (hopefully) retrieve the same encoded inputs back
- All images in dataset reduced to discrete set of vectors
- Mapping images to dictionary values - don't need to actually store these values, just the indices at which the values are located. Effectively mapping images to indices (integers).

Training loss fns:
- Reconstruction loss: MSE(x, output). Makes sure that the output matches the input
- Commitment loss: MSE(encoder_out, selected_keys) + MSE(selected_keys + encoder_out). Encourages the encoder output to approach/converge to the corresponding codebook entries. Also encourages the codebook to converge to the corresponding encoder output entries. A 2 way loss function that optimises both the encoder network and the codebook.

Auto-regressive image generation:
- Use auto-regressive modelling of the indices saved from the image mapping
- Sequential modelling of which index should come next (like an LLM modelling which word should come next)
- eg. PixelCNN, Transformer, RNN

Training with a pixel CNN prior:
- Encoded indices z are processed through convolution layers to give a predicted set of indices
- Every predicted index can only see the indices that come before it (apply a mask to all convolutional weights to ignore indices after the center index in the current convolution)


https://arxiv.org/pdf/1711.00937.pdf

https://keras.io/examples/generative/vq_vae/