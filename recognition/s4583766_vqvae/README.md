# Generative model using VQ-VAE
COMP3710 Pattern Recognition Report
Sophie Bates, 45837663.

## Project overview

### Dependencies
This is a python project that requires at least Python 3.11.x. Miniconda3 was used for package and dependency management. The dependencies (and their version numbers) required for running this project are as follows:
| **Dependency** | **Version** |
|---|---|
| pytorch | 2.0.1 |
| numpy | 1.25.0 |
| matplotlib | 3.7.1 |
| torchvision | 0.15.2 |
| scikit-image | 0.20.0 | 

The `.yaml` file [environment.yml](environment.yml) contains the conda environment used for this project, generated on a Linux OS (AlmaLinux release 8.8). To create the environment `conda-torch`, run the following command:
```bash
conda env create -f environment.yml
```

### Reproducing training and testing results
The entrypoint to the project is [train.py](train.py). To train the model, run the following command:
```bash
python3 train.py
```
This will train the model and create a new directory `/gen_imgs/x` where `x` was the date and time that the run started. This directory will contain the images generated in training, as well as the best model checkpoint (saved as a `.pth` file), and the training and validation losses and metrics (saved as `.png` files).

To evalute training results, the [predict.py](predict.py) script can be used. This script will load the best VQVAE model checkpoint and generate images from the test set. To run this script, run the following command, passing the path to the model checkpoint as an argument (the path should be relative to the `gen_img/` directory, i.e. ``):
```bash
python3 predict.py /path/to/model.pth
```
This will generate testing plots and images in a new directory. 

### File structure
The modules contained in this project are as follows:

| **Module** | **Description** |
|---|---|
| [dataset.py](dataset.py) | Loads the data and preprocesses it for use by the train loaders. |
| [modules.py](modules.py) | Core components of the model required for the pattern recognition task, includes VQVAE and GAN models. |
| [predict.py](predict.py) | Example usage of the trained model, generates results and provides visualisations. |
| [README.md](README.md) | This file! |
| [train.py](train.py) | Training script for the VQVAE and GAN models, including validation, testing, and saving of the model, and plotting losses and metrics observed during training. |

### Other notes
[Conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) were used to structure commit messages in this project. 


## Deep learning pipeline overview

1. [Data preparation](#1-data-preparation)
1. [Data pipeline](#2-data-pipeline)
1. [Model architecture](#3-model-architecture)
1. [Training procedure](#4-training-procedure)
1. [Testing procedure](#5-testing-procedure)
1. [Analysis](#6-analysis)

## 1. Data preparation

The data preparation methods are contained in the [dataset.py](dataset.py) file.

The dataset used for this project was the OASIS-3 dataset, from the Open Access Series of Imaging Studies (OASIS) [[1]](https://www.oasis-brains.org/). The dataset contains approximately 11,000 MRI scans of different brains, given as PNG images of dimension 256 x 256. To train the models, the OASIS dataset was split into training, testing, and validation datasets as follows:

| Dataset | No. images |
|---|---|
| Training | 9664 |
| Validation | 1120 |
| Testing | 544


These splits correspond to roughly 85%, 10%, and 5% of the dataset respectively. It was important that there were sufficient images in the testing set to ensure that the models were able to generalise to unseen data adequately. 

Shuffling of the training data was enabled in the data loader to ensure that the order of the images did not affect the training process. 

## 2. Data pipeline

No augmentation was performed to the images in the dataset, given it was of a sufficient size for VQVAE and GAN training. The only transforms required were to convert each image to a tensor, which is done in the OasisDataset class, in [dataset](dataset.py) module.

## 3. Model Architecture

The [modules.py](modules.py) module contains the model architecture for the VQVAE and GAN models.

### VQ-VAE 

The Vector Quantized Variational Autoencoder (VQ-VAE) is an extension of the regular auto-encoder architecture that contains an additional vector quantization layer between the encoder and decoder. The vector quantization layer is used to discretize the latent space, which allows for the model to learn a discrete latent representation of the data. 

The goal is to use the VQ-VAE as a generative model, where the latent space is used to generate novel images. This is achieved by imposing structure into the latent space. 

Previously, the latent space in a VAE was continuous, and the prior was a standard Gaussian. The VQ-VAE uses a discrete latent space, and the prior is a learned distribution (where the DCGAN comes in).

For this dataset, (1 channel etc. )
<!-- TODO -->

### DCGAN

The Deep Convolutional Generative Adversarial Network (DCGAN) is a generative model that uses convolutional layers to generate images.

## Model Architecture

Taken from [Neural Discrete Representation Learning paper](https://arxiv.org/pdf/1711.00937.pdf).

## 4. Training procedure
<!-- Hyper-parameters etc.  -->

The [train.py](train.py) module contains the training procedure. The training module contains two main functions: `train_vqvae` and `train_gan`. These functions are responsible for training the VQVAE and GAN models respectively.

### VQ-VAE training

The VQ-VAE is an implementation of the description provided the original paper on VQ-VAEs [[3]](#references). The core of the training functionality was adapted from [[2]](#references), including the hyper-parameters. The goal of the VQ-VAE training is to minimise the reconstruction error between a real image and the same image that has been fed through the VQ-VAE's encoder, vector quantizer, and then decoder.

#### Model parameters
* BATCH_SIZE = 32
* EPOCHS = 6
* EMBEDDINGS_DIM = 64  # Dimension of each embedding in the codebook (embeddings vector)
* N_EMBEDDINGS = 512  # Size of the codebook (number of embeddings)
* BETA = 0.25
* LEARNING_RATE = 1e-3

Loss function: Mean-squared error loss of the original image vs the reconstructed image. 

Optimizer: Adam, with learning rate = 1e-3. 

#### Training procedure
At each epoch, the the VQVAE model was trained by feeding real images through the entire model (encoder, vector quantizer, decoder). For each omage, the embedding and reconstruction losses were calculated and backpropagated through the model to update weights. 

#### Results
To visualise the training results, the following plots show a fixed batch of images before training:

<!-- insert file from resources/ -->

![Batch of training images before training](resources/epoch_0.png)

The same batch of images being passed through the VQ-VAE (after the first epoch):

![Reconstructed images (after 1 epoch)](resources/epoch_1.png)

After 6 epochs:

![Reconstructed images (after 6 epochs)](resources/epoch_6.png)

It is clear from these images that the reconstructions became clearer with training, however they did converge very quickly. A clearer comparison is shown below:

<p align="center">
  <img src="resources/epoch_0_single.png" width="30%" />
  <img src="resources/epoch_1_single.png" width="30%" /> 
  <img src="resources/epoch_6_single.png" width="30%" />
</p>

The following images show a single image comparison of the input image -> codebook embedding representation of the image -> and decoded reconstruction from the codebook:

<!-- ![Single image comparison](resources/codebook_comparison.png) 

![Single image comparison](resources/codebook.png) 

![Single image comparison](resources/decoded.png) -->

<p align="center">
  <figure>
    <figcaption style="max-width: 100%;">Figure 1: Comparison of input image, codebook embedding, and decoded reconstruction.</figcaption>
    <img src="resources/codebook_comparison.png" width="30%" />
    <img src="resources/codebook.png" width="30%" /> 
    <img src="resources/decoded.png" width="30%" />
  </figure>
</p>


The SSIM and MSE loss were also calculated for each epoch. The following plots show the results of these metrics:

<!-- TODO: plots here -->


### DCGAN training

The goal of this step is to train a DCGAN model that can generate the priors for the VQ-VAE. 

#### Model parameters
* LEARNING_RATE = 2e-4
* BATCH_SIZE = 256
* EPOCHS = 20

Loss function: binary cross entropy loss.

Optimizer: Adam, with betas = (0.5, 0.999)

#### Training procedure

The traing procedure for the DCGAN model was adapted from the [DCGAN tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) provided by PyTorch. In general, training a DCGAN employs elements from Game Theory to simultaneously train both the discriminator and generator models. At each epoch, the following steps are performed over each batch in the training set:
1. A batch of real images are fed through the discriminator, and the loss is calculated. 
1. Using a randomly sampled latent, the generator produces a batch of fake images.
1. The generated fake images are fed through the discriminator, and the loss is calculated. 
	1. The discriminator loss for each training step is the sum of the loss for the real and fake images: `log(D(x)) + log(1 - D(G(z)))`, where `D(x)` is the discriminator output for the real image, and `D(G(z))` is the discriminator output for the fake image.
	1. The generator loss for each training step is just the loss of the fake images: `log(D(G(z)))`. 
1. The losses for each model is backpropagated through the model, and the parameters are updated.

#### Results

The following plot shows the loss results from training the DCGAN model. The loss for the discriminator and generator are shown in blue and orange respectively.

TODO: plot goes here. 


## 5. Testing procedure
The [predict.py](predict.py) module contains the script for model evaluation, which calculates the Structural Similarity (SSIM) metrics over all training data. It loads in and evaluates any VQVAE model checkpoint. 

### VQ-VAE

#### Results

The output from evaluation was as follows:

```
SSIM mean: 0.7041
Min SSIM score: 0.5730
Max SSIM score: 0.7698
Number of images with SSIM >= 0.6: 535, 98.35%.
```

The plot of the SSIM scores is shown below:

![SSIM scores](resources/test_vqvae/test_ssim_scores.png)

The highest SSIM score, 0.7698, was observed for the following image:

<p align="center">
  <!-- <figure> -->
    <!-- <figcaption style="max-width: 100%;">Figure 1: Before and after encoding and reconstruction.</figcaption> -->
    <img src="resources/test_vqvae/best_recon_before.png" width="48%" />
    <img src="resources/test_vqvae/best_recon.png" width="48%" /> 
  <!-- </figure> -->
</p>


The lowest SSIM score, 0.5730, was observed for the following image:

<p align="center">
	<img src="resources/test_vqvae/worst_recon_before.png" width="48%" />
	<img src="resources/test_vqvae/worst_recon.png" width="48%" /> 
</p>

Overall, the SSIM scores were quite high, with 535 out of the 544 images (98.35%) in the dataset being above the miniminum threshold of 0.6. This indicates that the VQ-VAE model was able to reconstruct the images with a high degree of accuracy. Also, importantly, these scores showed that the model has decent generalisability and isn't overfitting, being only <!-- TODO --> lower than the training SSIM scores. 

## 6. Analysis

Results here from evaluation

## Future work

## Applications of this work
* Discretized = smaller images, so can be used for compression. 

## References
* [1] OASIS brain MRI dataset: https://www.oasis-brains.org/
* [2] VQ-VAEs: Neural Discrete Representation Learning: https://www.youtube.com/watch?v=VZFVUrYcig0.
* [3] Paper: *Neural Discrete Representation Learning*, Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu, 2017. https://arxiv.org/abs/1711.00937
* [4] DCGAN tutorial by Pytorch: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html.
* [Sonnet VQ-VAE implementation](https://github.com/google-deepmind/sonnet/blob/v1/sonnet/examples/vqvae_example.ipynb)

