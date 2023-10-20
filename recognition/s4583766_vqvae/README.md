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

To evalute training results, the [predict.py](predict.py) script can be used. This script will load the best model checkpoint and generate images from the test set. To run this script, run the following command, passing the path to the model checkpoint as an argument:
```bash
python3 predict.py --path /path/to/model.pth
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

The dataset used for this project was the OASIS-3 dataset, from the Open Access Series of Imaging Studies (OASIS) [[1]](https://www.oasis-brains.org/). The dataset contains approximately 11,000 png images that are MRI scans of different brains. To train the models, the OASIS dataset was split into training, testing, and validation datasets as follows:
* Training: 9664 images
* Validation: 1120 images
* Testing: 544 images

These splits correspond to roughly 85%, 10%, and 5% of the dataset respectively. It was important that there were sufficient images in the testing set to ensure that the models were able to generalise to unseen data adequately. 

Shuffling of the training data was enabled in the data loader to ensure that the order of the images did not affect the training process. 

## 2. Data pipeline

No augmentation was performed to the images in the dataset, given it was of a sufficient size for VQVAE and GAN training. The only transforms required were to convert each image to a tensor, which is done in the OasisDataset class, in [dataset](dataset.py) module.

## 3. Model Architecture

The [modules.py](modules.py) module contains the model architecture for the VQVAE and GAN models.

## 4. Training procedure
Hyper-parameters etc. 

The [train.py](train.py) module contains the training procedure. The training module contains two main functions: `train_vqvae` and `train_gan`. These functions are responsible for training the VQVAE and GAN models respectively.

## 5. Testing procedure
The [predict.py](predict.py) module contains the testing procedure, used to evaluate the models that were generated during training.

## 6. Analysis

Results here from evaluation

## Future work
## References
* [1] https://www.oasis-brains.org/
* [Sonnet VQ-VAE implementation](https://github.com/google-deepmind/sonnet/blob/v1/sonnet/examples/vqvae_example.ipynb)
* [Video](https://www.youtube.com/watch?v=VZFVUrYcig0)