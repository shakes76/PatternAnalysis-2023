# Generative model using VQ-VAE
COMP3710 Pattern Recognition Report
Sophie Bates, 45837663.

## Project notes
[Conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) were used to structure commit messages. 

## Usage
### Dependencies
This is a python project that requires at least Python 3.11.x. Miniconda3 was used for package and dependency management. The dependencies (and their version numbers) required for running this project are as follows:
| **Dependency** | **Version** |
|---|---|
| pytorch | 2.0.1 |
| numpy | 1.25.0 |
| matplotlib | 3.7.1 |
| torchvision | 0.15.2 |
| scikit-image | 0.20.0 | 

[environment.yml](environment.yml) contains the conda environment used for this project. To create the environment `conda-torch`, run the following command:
```bash
conda env create -f environment.yml
```

### Training and testing
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


## File structure
The modules contained in this project are as follows:

| **Module** | **Description** |
|---|---|
| [dataset.py](dataset.py) | Loads the data and preprocesses it for use by the train loaders. |
| [modules.py](modules.py) | Core components of the model required for the pattern recognition task, includes VQVAE and GAN models. |
| [predict.py](predict.py) | Example usage of the trained model, generates results and provides visualisations. |
| [README.md](README.md) | This file! |
| [train.py](train.py) | Training script for the VQVAE and GAN models, including validation, testing, and saving of the model, and plotting losses and metrics observed during training. |

## Deep learning pipeline overview

1. [Data preparation](#1-data-preparation)
1. [Data pipeline](#2-data-pipeline)
1. [Model architecture](#3-model-architecture)
1. [Training procedure](#4-training-procedure)
1. [Testing procedure](#5-testing-procedure)
1. [Analysis](#6-analysis)

## References
