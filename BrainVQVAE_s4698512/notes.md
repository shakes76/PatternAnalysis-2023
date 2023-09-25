## VQ VAE Notes

Differs from VAE in two main ways

1. Encoder network now outputs discrete rather than continuous codes
2. Prior model is learned rather than being static

AutoEncoders and Variational AutoEncoders

AutoEncoder simply projects input down onto Latent Space
Then reprojects (upscales) it back to (hopefully) get back to the same image

A Variational AutoEncoder (VAE) introduces a stochastic nature to the AE so that it
is now capable of generating new samples rather than just reproducing existing samples from the latent space

In a Vector Quantised Variational AutoEncoder (VQ-VAE), the continuous values learned by the encoder
are discretised into a set of finite vectors. These are drawn from a pre-defined codebook. The codebook
is like a dictionary of possible features or patterns.

KL Divergence
Trying to maximise elbow
=
minimising deconstruction loss

KL Divergence minimises difference betweeen posterior and Gaussian Prior.
Minimises loss

Generative Model

Want to plug in some latent vector and output an image

Need to impose structure into latent space
Ensure it's continuous and meaningful
VAE does this

How about VQ-VAE
Has a discrete latent space
Prior still Gaussian is nice
