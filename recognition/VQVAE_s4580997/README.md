# VQVAE on the ADNI Brain Data Set

## Table of Contents
- [Background](#background)
    - [Problem Description](#problem-description)
    - [Dataset: ADNI](#dataset-adni)
    - [Architecture](#architecture)
        - [VQVAE](#vqvae)
        - [Prior](#prior)
- [Dependencies](#dependencies)
- [Documentation](#documentation)
    - [dataset.py](#datasetpy)
        - [Loader](#class-loader)
        - [Dataset](#class-dataset)
        - [ModelLoader](#class-modelloader)
        - [ModelDataset](#class-modeldataset)
    - [modules.py](#modulespy)
        - [ResidualLayer](#class-residuallayernnmodule)
        - [ResidualBlock](#class-residualblocknnmodule)
        - [Encoder](#class-encodernnmodule)
        - [VectorQuantiser](#class-vectorquantisernnmodule)
        - [Decoder](#class-decodernnmodule)
        - [Discriminator](#class-discriminatornnmodule)
        - [Generator](#class-discriminatornnmodule)
        - [GAN](#class-gannnmodule)
    - [predict.py](#predictpy)
        - [Predict](#class-predict)
    - [train.py](#trainpy)
        - [Trainer](#abstract-class-trainerabc)
        - [TrainVQVAE](#class-trainvqvaetrainer)
        - [TrainGAN](#class-traingantrainer)
- [Usage](#usage)
- [Results](#results)
    - [Training](#training)
    - [Validation](#validation)
    - [Generation](#generation)
- [Conclusion](#conclusion)
- [References](#references)

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
The Vector Quantized Variational Autoencoder (VQ-VAE) is a variant of the standard Variational Autoencoder (VAE) model. The VAE model operates in a continuous latent space, with sampling via a Gaussian distribution [2]. The VQVAE adds a vector quantisation layer to instead learn a discrete latent representation [1].

Image

The VQ-VAE model consists of an encoder, decoder and an added vector quantisation layer. The encoder network parameterises the distribution of the data to convert it to an encoding vector, with each dimension a learned attribute of the data [5]. The vector quantiser then discretises this encoding vector from continuous to produce a discrete latent representation. The decoder then reconstructs the data from the discrete latent representation. The vector quantiser is trained to minimise the distance between the input and output of the encoder and decoder, and the encoder and decoder are trained to minimise the distance between the input and output of the vector quantiser [1]. 

The loss function for training the VQVAE is comprised of the embedding loss and reconstruction loss.

#### Prior
The initial model for the VQ-VAE employed an autoregressive encoder via a PixelCNN for the generation of images. The generative model used as the prior is a generative adversarial network, or simply GAN/DCGAN, which can be used to take the encoded indices from the VQ-VAE to generate images based on the discrete space representation of the codebook.

Image

The GAN network trains a generator and discriminator in competition. The discriminator is a binary classification network, which classifies an input/image  as being derived from the generator distribution or data distribution. The generator then aims to maximise the probability of the discriminator incorrectly classifying the output [4].



## Dependencies
The package dependencies used for the project are:
| **Dependency** | **Version** |
|---|---|
| Python | 3.7.16 |
| pytorch | 1.13.1 |
| torchvision | 0.14.1 |
| numpy | 1.21.5 |
| matplotlib | 3.5.3 |
| skimage | 0.22 |

## Documentation
Documentation for all classes, documenting their associated public methods and parameters is provided below. The implementation of the software flow is class-based, and hence, is not executable directly as a script. Executable implementation is provided in `main.py`, with all parameters defined in `utils.py`.

### `dataset.py`
#### Class `Loader()`
Abstract class for implementing the dataset object.

**Parameters**
- __batch_size, 32__ -> Size of the batch for processing
- __path__ -> Directory path of the dataset
- __fraction, 1.0__ -> How much of the dataset to use to create the loader
- __transform__ -> Transform to apply to the data

**Methods**
`load() -> None`
Load the dataset from the ImageFolder at the given path and store it.

`get() -> DataLoader`
Return the loaded dataset.

`unloaded() -> bool`
Verify that the dataset has been loaded.

`variance() -> float`
Compute the variance of the dataset and store it. Once computed, it does not need to be recalculated.

#### Class `Dataset()`
**Parameters**
- __batch_size, 32__ -> Size of the batch for processing
- __path__ -> Directory path of the dataset for both test and train splits
- __fraction, 1.0__ -> How much of the dataset to use to create the loader

**Methods**
`load_train() -> None`
Loads the data for the training set.

`get_train() -> DataLoader`
Returns the training dataset.

`load_test() -> None`
Loads the data for the testing set.

`get_test() -> DataLoader`
Returns the testing dataset.

`train_unloaded() -> bool`
Check if the training set has not yet been loaded.

`train_var() -> float`
Get the variance of the training set.

`test_unloaded() -> bool`
Check if the testing set has not yet been loaded.

`test_var() -> float`
Get the variance of the testing set.

#### Class `ModelLoader()`
**Parameters**

**Methods**

#### Class `ModelDataset()`
**Parameters**

**Methods**

### `modules.py`
#### Class `ResidualLayer(nn.Module)`
Reusable residual layer for use in residual stacks/blocks.
**Parameters**
__in_channels__ Number of input channels
__n_hidden__ Number of output channels (generally equal to number of input)
__n_residual__ Number of residual hidden layers


#### Class `ResidualBlock(nn.Module)`
**Parameters**
- __dim_in__ Dimension of the tensor input
- __dim_hidden__ Dimension of the hidden layers
- __dim_residual__ Dimension of the residual hidden layer
- __n_residuals__ Number of residual layers

#### Class `Encoder(nn.Module)`
**Parameters**
- __in_channels__ Number of input channels
- __n_hidden__ Number of hidden layers
- __n_residual__ Number of residual hidden layers
- __out_channels__ Number of channels of output

#### Class `VectorQuantiser(nn.Module)`
**Parameters**
- __n_embeddings__ Number of embeddings in the codebook
- __dim_embedding__ Dimension of each embedding
- __beta__ Commitment cost of loss

#### Class `Decoder(nn.Module)`
**Parameters**
- __in_channels__ Number of input channels.
- __n_hidden__ Number of hidden layers
- __n_residual__ Number of residual hidden layers
- __out_channels__ Number of channels for the output

#### Class `VQVAE(nn.Module)`
Class which layers the encoder, vector quantiser and decoder.
**Parameters**
- __channels__ Number of input channels
- __n_hidden__ Number of hidden layers
- __n_residual__ Number of residual hidden layers
- __n_embeddings__ Number of embeddings in the codebook
- __dim_embedding__ Dimension of each embedding
- __beta__ Commitment cost term for vector quantizer

#### Class `Discriminator(nn.Module)`
Discriminator is used as a binary classifier to detect the quality of the generator. It will output the probability of the image being real or fake.

**Parameters**
- __channels__ Number of input channels on the image
- __img_size__ Size of the image

#### Class `Generator(nn.Module)`
The generator will create images of the selected size from the random latents.
**Parameters**
- __latent_size__ Latent size of input vector
- __channels__ Number of channels for the output


#### Class `GAN()`
GAN class for containing the generator and discriminator models to ensure that the parameters passed to the models are aligned.
**Parameters**
- __channels__ Number of input channels
- __latent_dim__ Size of noise
- __img_size__ Size of image output, assumed square

### `predict.py`
#### Class `Predict()`
The Predict class is specific to the VQVAE with Prior model. It handles the generation of images and provides performance metrics.
**Parameters**
- __vqvae__ A trained VQVAE model
- __gan__ A trained GAN
- __dataset__ Dataset 
- __device, 'cpu'__ Device to perform the methods on
- __savepath, './models/predictions/'__ The directory path to save the outputs to by default
- __img_size, 64__ The size of the images to be generated

**Methods**


### `train.py`
#### Abstract Class `Trainer(ABC)`
This super class is provided to define the required methods and initialise parameters for any future train classes.

**Parameters**
- __model: nn.Module__ The module to be trained
- __dataset: Dataset__ Dataset object containing train data
- __lr, 1e-3__ The learning to use
- __wd, 0__ Weight decay to use
- __epochs, 10__ Number of epochs to train for
- __savepath, './models/vqvae'__ The default save directory for outputs and the model

**Methods**
`train() -> None`
Abstract method to implement to define the train behaviour.

`validate() -> None`
Abstract method to implement to define the validate behaviour.

`plot() -> None`
Abstract method to define how to plot any results.

#### Class `TrainVQVAE(Trainer)`
Trainer specific for the defined VQVAE module. It uses the Adam optimiser by default, and error is defined as the embedding loss summed with the mean-squared error of the model output.

**Parameters**
- __model: nn.Module__ The module to be trained
- __dataset: Dataset__ Dataset object containing train data
- __lr, 1e-3__ The learning to use
- __wd, 0__ Weight decay to use
- __epochs, 10__ Number of epochs to train for
- __savepath, './models/vqvae'__ The default save directory for outputs and the model

**Methods**
`train() -> None`
Training is done in batches from the dataset. The loss is saved to the model via `self.losses` for future plotting. This must be run to train the model.

`plot(save = True) -> None`
Implementing of the abstract plot method. It will plot the losses at each iteration computed during training. If save is true, then the figure will be saved to the savepath.

`save(newpath = None) -> None`
Method to save the state dictionary of the model network. The save path can be overriden using newpath, else it will be saved to the directory of savepath.

#### Class `TrainGAN(Trainer)`
Trainer specific for the defined GAN module. It uses the Adam optimiser by default for both generator and discriminator. The criterion used is BCE loss, with the generato

**Parameters**
- __model: nn.Module__ The module to be trained
- __dataset: Dataset__ Dataset object containing train data
- __lr, 1e-3__ The learning to use
- __wd, 0__ Weight decay to use
- __epochs, 10__ Number of epochs to train for
- __savepath, './models/gan'__ The default save directory for outputs and the model

**Methods**
`train() -> None`
Training is done in batches from the dataset. Two models are 

`plot(save = True) -> None`
Implementing of the abstract plot method. It will plot the losses at each iteration computed during training. If save is true, then the figure will be saved to the savepath.

`save(newpath = None) -> None`
Method to save the state dictionary of the model network. The save path can be overriden using newpath, else it will be saved to the directory of savepath.

## Usage
The code is bundled using the `utils.py` file into `main.py` which is an directly executable script. All files other than these two define the architecture, training, testing, dataset and prediction segments of the problem.

Parameters can be edited in `utils.py` to modify the architecture or system. However, components of the code can also be modified to produce custom networks and/or change the models.

For the dataset, create a new Dataset class with the path to the dataset, and the fraction of the dataset to use. This data can be in image folders with labels if desired.
```console
dataset = Dataset()
```

The model can then be created from any child class of `nn.class`. Using the selected `VQVAE()` and `GAN` classes:
```
vqvae = VQVAE()
```

The trainer can then be defined as a child class of `Trainer()` in `train.py`. As we are using the VQVAE model, we will define it:
```
vqvae_trainer = TrainVQVAE(vqvae, dataset)
gan_trainer = TrainGAN(gan, dataset)
```
We can then train the model using the trainer and save the state dictionary containing the network weights to a selected directory and filename:
```console
vqvae_trainer.train()
vqvae_trainer.save()
gan_trainer.train()
gan_trainer.save()
```

With the model trained, predictions can be performed using a child class of `Predict`. The predict class is implemented specifically to use the prior model (GAN) to generate images using the codebook of the VQVAE. Usage is as follows:
```
predict = Predict(vqvae, gan, dataset)
predict.generate()
```

It is highly recommended to use the `main.py` file to execute the code as desired, with the parameters predefined in `utils.py`.

## Results
### Training


### Validation

### Generation
Images can be generated

## Conclusion

## References
[1] A. v. d. Oord, O. Vinyals, and K. Kavukcuoglu, “Neural Discrete Representation Learning,”
arXiv:1711.00937 [cs], May 2018, arXiv: 1711.00937. [Online]. Available: http://arxiv.org/abs/1711.00937

[2] Laskin, Misha (2019), Vector Quantized Variational Autoencoder. Available: https://github.com/MishaLaskin/vqvae

[3] Team, K. (2021). Keras documentation: Vector-Quantized Variational Autoencoders. [online] Keras.io. Available: https://keras.io/examples/generative/vq_vae

[4] https://arxiv.org/pdf/1406.2661v1.pdf

[5] https://www.jeremyjordan.me/variational-autoencoders/