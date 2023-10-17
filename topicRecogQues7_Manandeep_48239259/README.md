# Siamese Network for Alzheimer's Disease Detection

## Overview

This project demonstrates the use of a Siamese Network for Alzheimer's Disease detection using deep learning techniques. The Siamese Network is designed to compare pairs of medical images and determine whether they belong to the same class (e.g., Alzheimer's Disease or Cognitive Normal). The goal is to achieve an accuracy of approximately 0.8 on the test set. This README provides an overview of the project structure, how to use it, and what each component does.

## Model Architecture

The Siamese Network architecture used in this project consists of the following components:

- Convolutional Neural Network (CNN) layers for feature extraction.
- Fully connected layers for similarity scoring.
- Contrastive loss function for training.

The network is designed to compare pairs of medical images and determine their similarity based on feature extraction and loss computation.


## Data Preprocessing

Data preprocessing includes the following steps:

- Loading the medical image dataset using PyTorch's torchvision datasets.
- Image transformations, such as resizing and converting to tensors.
- Pairwise image comparison to create positive and negative training samples for the Siamese Network.
- Inversion of images for better feature extraction.

## Training and Validation Evidence

The model was trained for a specified number of epochs, and the training progress was monitored. Here's some evidence of the training:

- Number of Epochs: 6
- Training Loss: Graphs and statistics can be visualized in the 'Training Loss' section of this README.



## Project Structure

The project is organized into four main components:

1. **modules.py**: Contains the source code for the components of the Siamese Network, including the network architecture and the contrastive loss function.

2. **dataset.py**: Defines the data loader for loading and preprocessing the medical image data.

3. **train.py**: Contains the code for training the Siamese Network, including data loading, model training, and saving the trained model.

4. **predict.py**: Demonstrates the usage of the trained model for image classification, including classifying test images and calculating accuracy.
