# VQVAE with PixelCNN Prior for ADNI Brain Data Set Generation
## Description
Variational autoencoders (VQVAE) are a class of generative models proficient in reconstructing and generating high-dimensional data from lower-dimensional embeddings. In our unique implementation, we incorporate PixelCNN as a prior for the latent codes, which enhances the quality and diversity of generated MRI brain scans from the ADNI dataset. Our combined approach not only reconstructs the intricate details and structures of brain scans but also achieves a Structured Similarity (SSIM) of over 0.6.

## How it works
### VQVAE:
VQVAE employs an encoder that maps input images to lower-dimensional latent codes and a decoder that reconstructs images from these codes. A distinguishing feature of VQVAE is the use of a vector quantizer, which discretizes the continuous latent space, allowing for a more effective training and reconstruction process.

The image below depicts the architecture of the VQVAE <br>
![VQVAE architecture](assets/vqvae_architecture.png)


## Data Pre-processing
The ADNI brain data set was pre-processed by:

- Normalizing the pixel values between 0 and 1.
- Resizing the images to a consistent size for input to the VQVAE.
- Training-Validation Split: 80-20

The training set accounts for 80% of the train directory from the ADNI dataset. The validation set, 20%, ensures hyperparameter tuning without overfitting. The test set comes from a separate test directory from the ADNI dataset.

## Visualization
### SSIM VQVAE train plot
The graph below plots the SSIM value throughout training the VQVAE <br>
![SSIM plot](assets/ssim_plot.png)

The graph below plots the training loss throughout training the VQVAE <br>
![VQVAE loss plot](assets/VQVAE_loss_plot.png)


## Dependencies
- PyTorch 1.10.0
- torchvision 0.11.1
- pytorch-msssim
- sklearn

## Example Inputs, Outputs, and Generated Images
### Reconstructed MRI Image by VQVAE
The image below is the reconstructions from training at the last (10th) epoch. The top images are the original brain scans and the bottom are the reconstructions. <br>
![Train Reconstructions](assets/train_reconstructions.png)


The image below is the reconstructions from testing. The top images are the original brain scans and the bottom are the reconstructions. <br>
![Test Reconstructions](assets/test_reconstructions.png)

### Generated: New MRI Images
The image below is the generated images from the samples fetched from the distribution. <br>
![Generated Images](assets/generated_images.png)



