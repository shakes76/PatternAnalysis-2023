# VQVAE on the ADNI Brain Data Set

## Background
### Problem Description
The selected problem is to develop a generative model on the ADNI brain dataset using a VQVAE, to learn the latent space representation and thus produce realistic images.

### Dataset: ADNI
The models are trained on the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset. This consists of MRI of the brain for selected patients, with the resulting data labelled as Cognitive Normal (NC) and those with Alzheimer's Disease (AD). The dataset is already separated into a train and test split as follows:

- 21520 images in the training split
    - 10400 images of Alzheimer's Disease (AD)
    - 11120 images of Cognitive Normal (NC)
- 9000 images in the test split
    - 4460 images of Alzheimer's Disease (AD)
    - 4540 images of Cognitive Normal (NC)

Images were scaled to 128 by 128 for interaction with the models used. No augmentation was used as the dataset is of a suitable size.

### Architecture
#### VQVAE

Very efficient at representing a latent space in imaging

The loss function for training the VQVAE

#### Priors
A prior model was used to provide more controlled generation to predict the next code from a given sequence. This was required as randomly sampling from the embedded space resulted in poor images, as the sequences of codes were not meaningful. 


## Dependencies
The package dependencies used for the project are:
| **Dependency** | **Version** |
|---|---|
| Python | 3.7.16 |
| pytorch | 1.13.1 |
| torchvision | 0.14.1 |
| numpy | 1.21.5 |
| matplotlib | 3.5.3 |
| skimage | 3.5.3 |

## Documentation
### 


## Results
### Training

### Validation / Testing

### Generation
Images can be generated

## References
[1] A. v. d. Oord, O. Vinyals, and K. Kavukcuoglu, “Neural Discrete Representation Learning,”
arXiv:1711.00937 [cs], May 2018, arXiv: 1711.00937. [Online]. Available: http://arxiv.org/abs/1711.00937

[2] Laskin, Misha (2019), Vector Quantized Variational Autoencoder. Available: https://github.com/MishaLaskin/vqvae

[3] Team, K. (2021). Keras documentation: Vector-Quantized Variational Autoencoders. [online] Keras.io. Available: https://keras.io/examples/generative/vq_vae

