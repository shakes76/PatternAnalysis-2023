# COMP3710 Project: Siamese network Classifier for ADNI data
**Student Number:** 48240639
**Name:** Aniket Gupta 
**Description:** The Submitted Project contains the files for COMP3710 project report for the Year 2023

## Table of Content
1. [Introduction](#1-introduction)
2. [Project structure](#2-project-structure)
3. [Reproducibility](#3-reproducibility)

## 1. Introduction
The main aim of this project is to create a network that can classify Alzheimers disease by distinguishing between patients and those, with AD. To accomplish this we will use the ADNI brain dataset. Target an accuracy level of 0.8.

Siamese Networks are a type of network architecture designed for tasks involving image comparison and similarity verification. In this architecture there are two subnetworks called "subnetworks each processing its own input and generating a feature vector as output. These feature vectors are. Merged for further analysis or comparison purposes.


## 2. Project structure
1. ```modules.py``` This code defines a Siamese Network architecture for image similarity comparison.
The network is built upon the ResNet-18 architecture with modifications to handle
grayscale images. It includes custom weight initialization and forward pass methods.

2. ```dataset.py``` This code defines a custom dataset class, ADNIDataset, for loading and processing
ADNI dataset images for use in Siamese Network training and testing. It also provides
functions to get train and test datasets from a specified data path.and 20% validating. This also inculude custom dataloader to handle PairDataset for Siamese model.
3. ```train.py``` This script is used to train a Siamese Network model on a dataset, with support for validation.
It includes training and validation loops, model saving, and TensorBoard logging.
4. ```predict.py``` This script is used to test a Siamese Network model on a dataset. It loads a trained Siamese Network model, 
evaluates its performance on a test dataset, and reports the accuracy and loss.


## 3. Reproducibility
This project can be replicated with confidence as it incorporates a deterministic algorithm for the convolutional layer and consistently sets a seed whenever random variables are involved.

### 3.1. Dependencies

- Python 3.10.12: Python is the primary programming language used in this project. Version 3.10.12 is specifically required to ensure compatibility with the project's code and libraries.

- Pytorch 2.0.1: PyTorch is a popular deep learning framework that is essential for building and training neural networks. Version 2.0.1 is used to make use of the latest features and bug fixes.

- Pillow 9.4.0: Pillow is a Python Imaging Library that adds image processing capabilities to your Python interpreter. Version 9.4.0 is used for working with images and data preprocessing in the project.

- torchvision 0.15.2: TorchVision is a PyTorch library that provides various utilities and datasets for computer vision tasks. Version 0.15.2 is used to access pre-trained models and datasets for the project.

These dependencies are crucial for running the project successfully, and the specified versions are recommended to ensure compatibility and avoid potential issues.

