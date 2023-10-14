# OASIS Brain Image Generation with VQ-VAE

## Description:
This repository provides an implementation of the Vector Quantized Variational AutoEncoder (VQ-VAE) to generate brain images based off the OASIS dataset. The objective is to create a generative model that produces reasonably clear images of the brain and achieves a Structural Similarity (SSIM) index of over 0.6. This work aims to offer a reliable method for generating high-quality representations of brain images, which can be beneficial for various neuroscience and medical applications.

## How it Works:
The VQ-VAE is a type of Variational AutoEncoder that incorporates vector quantization for its discrete representation. This allows for a distinct encoding of the input data, leading to more effective generation of images. This VQ-VAE model is trained using the OASIS brain dataset, specifically, the brain slices in PNG format. During training, the model learns to compress and reconstruct the brain images with high fidelity. In addition to metric evaluations, images were also generated during training to visually compare and ensure they were reasonably clear. The quality of the generated images is evaluated using the SSIM index, which measures the structural similarity between the original and reconstructed images.

## Visualisation:
### Architecture

### Training
![losses and ssim scores from training loop](./models6/loss_ssim_plot.png)

### First generation in training

### Final generation based off model parameters
![generated image based off model parameters](./models6/best_model_sample.png)

## Dependencies:
- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- skimage


## Reproducibility: -- fix this!!!!!
To ensure reproducible results, the model's random seeds for Pytorch and other libraries are set. However, some variability might arise due to GPU-specific operations or non-deterministic operations in the libraries used. It's recommended to run the algorithm on the same hardware and software configuration for consistent outcomes.

## Example Inputs and Outputs:
### Input: 
Brain slice images from the OASIS dataset.
### Output: 
A batch of generated brain slice images from the trained VQ-VAE model.
### Plots: 
Generated images based on the best saved model parameters. Additionally, training and validation losses and validation SSIM scores plotted over epochs.

## Pre-processing:
The brain slice images from the OASIS dataset are loaded as grayscale images and normalized to the range [0, 1]. This normalization helps the neural network converge faster and achieve better performance. No other specific pre-processing steps have been applied.

## Data Splits Justification:
The data is divided into three sets: training (80%), validation (10%), and testing (10%). The training set is the largest to ensure the model has enough data to learn effectively. The validation set assists in hyperparameter tuning based on SSIM performance. The test set is reserved to evaluate the final model's performance on unseen data. This 80-10-10 split is commonly used in machine learning to ensure a balance between model training and evaluation while preventing overfitting. The data loading was performed in the dataset.py module, where the dataset images were split up in this 80-10-10 ratio in the train, validate, and testing directories accordingly. 

## Code structure:
**modules.py**
- Contains helper functions and main VQVAE model classes, including the encoder, decoder, and the vector quantization process.
**dataset.py**
- Sets up data preprocessing for the OASIS brain dataset and initializes data loaders for training, validation, and testing splits using a batch size of 32 and grayscale normalization.
**train.py**
- Main training loop for the VQVAE on the OASIS dataset. This script handles the training process, validation, and testing of the VQ-VAE model. Model checkpoints are saved based on improvements in the combined metric of SSIM and reconstruction loss.
**predict.py**
- This script reads the best epoch from a saved model parameters file, loads the corresponding pre-trained VQVAE model, and uses it to generate and save an image sample from the test data. 

