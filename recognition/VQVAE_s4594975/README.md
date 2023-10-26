# VQVAE and Pixel CNN - Generative Model of OASIS Dataset
AIM: The aim of this project is to develop a generative model for one of the specified medical imaging datasets (OASIS brain, ADNI brain, or OAI AKOA knee) using a VQVAE (Vector Quantized Variational Autoencoder) with a PixelCNN (Pixel Convolutional Neural Network) to produce "reasonably clear" medical images with a Structured Similarity Index (SSIM) score exceeding 0.6. The project involves leveraging advanced deep learning techniques to create high-quality, structured, and interpretable medical images, which can be invaluable for medical diagnosis, research, and analysis.


## OASIS DATASET

The OASIS dataset, which stands for "Open Access Series of Imaging Studies," is a collection of neuroimaging and clinical data designed for research in neurodegenerative diseases, particularly Alzheimer's disease. It is a valuable resource for researchers, clinicians, and scientists interested in studying brain health, dementia, and related conditions.

Traning Images - 9,664 
Test Images - 544
Validation Images - 1120

## VQVAE
A VQVAE, or Vector Quantized Variational Autoencoder, is a type of neural network architecture used in the field of deep learning and generative modeling. The VQ-VAE operates using a discrete latent space, which is represented as a discrete codebook. The encoder part of the model characterizes this latent space as a categorical distribution. The codebook is established by converting the continuous embeddings and the encoded outputs into discrete code words. These discrete code words are subsequently fed into the decoder. The decoder is then trained to produce reconstructed samples based on these discrete code words. 

VQVAE -> 3 parts. Encoder, latent space and decoder.

"Architecture picture"

A VQVAE (Vector Quantized Variational Autoencoder) differs from a VAE (Variational Autoencoder) primarily in the nature of their latent space representations. In a VAE, the latent space is continuous and probabilistic, allowing for smooth data generation with continuous variations, while a VQVAE utilizes a discrete and quantized latent space achieved by mapping the continuous latent space into discrete codes.

## PIXEL CNN
PixelCNN is a type of generative model designed for generating images, particularly pixel by pixel. It models the conditional distribution of each pixel in an image given the previous pixels. It's capable of generating high-quality, highly structured images and can be used for various image generation tasks.

VQVAE runs alongside pixelcnn which trains to generate encodings. After the training, the pixelcnn is used to generate encodings that aren't exposed to the VQVAE. These are then decoded by the decoder layer in the VQVAE.

## DATASET - PREPROCESSING
Preprocessed dataset was used which contained train, test and validation images. The dataset was normalised to be in the range [-0.5 to 0.5] and pixel size was downsampled to 80x80 from 256x256 to compile faster and less memory resources.

## TRAINING 
### VQVAE
### PIXEL CNN
## RESULTS
