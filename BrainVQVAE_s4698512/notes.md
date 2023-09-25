## VQ VAE Notes

Differs from VAE in two main ways

1. Encoder network now outputs discrete rather than continuous codes
2. Prior model is learned rather than being static

AutoEncoders and Variational AutoEncoders

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
