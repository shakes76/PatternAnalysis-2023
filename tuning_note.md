# Tuning note


## VAE-GAN

To sample clear image:
* learning rate 2e-4 is good enough.
* If we don't care kld loss: 
  1. Remove Auxiliary (that is train decoder and discriminator by sample image)
  2. KLD loss term (grad adapt included) should be 1e-6

## VQVAE

LPIPS in discriminator is a bad idea. difference loss will keep growing if LPIPS added.