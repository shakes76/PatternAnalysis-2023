# **Implementing the VQVAE model on the OASIS-1 brain dataset**

## Background on VQVAE

### What is it?
VQ-VAE stands for Vector Quantized Variational Autoencoder. It is one type of variational autoencoder amongst others and uses vector quantisation to obtain discrete latent representation <sup>[2]</sup>. The overall model architecture can be seen in the image below.

![vqvaemodel](vqvae_model_architecture.png)

### Why it is effective?
Generally for VAE's, "the latent space is continuous and is sampled from a Gaussian distribution." <sup>[3]</sup> Whereby VQVAE's "operate on a discrete latent space, making the optimization problem simpler."<sup>[3]</sup> It does this by keeping a discrete codebook which discretizes the distance between continuous embeddings and encoded outputs. <sup>[3]</sup>

## Background on the OASIS-1 brain dataset
The OASIS-1 dataset consists of Cross-sectional MRI Data in young, middle aged, nondemented and demented older adults. This dataset arose from 416 subjects aged from 18 - 96 with over 434 MR sessions <sup>[1]</sup>.

## Background on DCGAN

### What is it?
The Deep Convolutional Generative Adversarial Network (DCGAN) is a type of CNN that serves as a way to generate new images after training generator and discriminator networks. <sup>[4]</sup> The overall model architecture of a DCGAN can be seen in the image below.

![dcgan](dcgan_model_architecture.png)

### Why is it effective?
A DCGAN was chosen over a regular GAN for this task due to it's improved performance for image generation as a result of its deep convolutional architecture. A DCGAN implements convolutional and convolutional-transpose layers in its generator and discriminator networks respectively. <sup>[4]</sup> Furthermore, DCGAN's have also been found to be more stable during training and have higher image quality when compared to regular GAN's.

## Preparing Datatsets
All dataset classes are created in the *dataset.py* file. There are two classes implemented for downloading the OASIS dataset from a specified directory, performing the necessary transforms and creates functions to be called when requiring either the train, validation or test dataloaders. This file also includes a class that handles the dataloader required for the DCGAN. This class takes in a trained VQVAE model, and returns the encodings of original OASIS training images. 

The following image shows an example of the original OASIS images that was used as training images for the VQVAE model. 

![original_images](Original_OASIS_images.png)

## Creating Models
Both the VQVAE and DCGAN models are implemented as classes within the *modules.py* file. For the VQVAE model, there are classes that create residual blocks(from residual layers), an encoder, a decoder, a vector quantizer and a class that combines all necessary components to form the VQVAE model. For the DCGAN, there is a Generator class and Discriminator class, as well as a function for initialising weights for these networks. 

## Training Procedure

### Training VQVAE
There are three main files involved for the training of the VQVAE and DCGAN. The *train.py* file calls classes from the *train_VQVAE.py* and *train_DCGAN.py* files for training each individual model. The *train_VQVAE.py* file includes a class with a function for training the VQVAE model, plotting the training reconstruction losses and saving the model to the current working directory. It also includes a function for validation testing after training the model. The model is trained on only 3 epochs as it was found to output sufficient reconstructions. The training and validation reconstruction loss plots can be seen below. 

![train_recon_loss_plot](reconstruction_err_train.png)

![val_recon_loss_plot](reconstruction_err_validate.png)

### Training DCGAN
Following the training of the VQVAE model, the DCGAN was trained on the encodings of the trained OASIS images. An example of a the encodings that formed the training set for the DCGAN can be seen below.

![vqvae_encoding_example]()

## Results


## Dependencies




# References
[1] www.oasis-brains.org. (n.d.). OASIS Brains - Open Access Series of Imaging Studies. [online] Available at: https://www.oasis-brains.org/.

[2] A. v. d. Oord, O. Vinyals, and K. Kavukcuoglu, “Neural Discrete Representation Learning,”
arXiv:1711.00937 [cs], May 2018, arXiv: 1711.00937. [Online]. Available: http://arxiv.org/abs/1711.00937

[3] Team, K. (2021). Keras documentation: Vector-Quantized Variational Autoencoders. [online] Keras.io. Available at: https://keras.io/examples/generative/vq_vae/#:~:text=In%20standard%20VAEs%2C%20the%20latent [Accessed 18 Sep. 2023].

[4] Radford, A., Metz, L. and Chintala, S. (n.d.). UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS. [online] Available at: https://arxiv.org/pdf/1511.06434v2.pdf.