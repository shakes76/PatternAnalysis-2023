# Alzheimer Image Classification with Vision Transformer (ViT) 

## Overview
This project is an implementation of a Vision Transformer (ViT) for image classification. The code includes components for loading, preprocessing, and training a ViT model on ADNI dataset. Additionally, it allows you to make predictions and provides metrics for model evaluation.

## Table of Contents

- [ADNI Dataset](#adni-dataset)
- [Project Structure](#project-structure)
  - [Files](#files)
- [Vision Transformer (ViT) Model Design](#vision-transformer-vit-model-design)
  - [Introduction](#introduction)
  - [Model Architecture](#model-architecture)
  - [Hyperparameters](#hyperparameters)
  - [Training](#training)
  - [Results](#results)
- [Usage](#usage)
- [Requirements](#requirements)

## ADNI Dataset
The Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset is a valuable and widely used collection of medical imaging and clinical data aimed at advancing our understanding of Alzheimer's disease. Comprising a comprehensive range of neuroimaging modalities, including MRI and PET scans, as well as clinical and cognitive assessments, the ADNI dataset has played a pivotal role in enhancing research related to neurodegenerative diseases. This dataset not only facilitates the identification of biomarkers associated with Alzheimer's disease but also promotes the development of innovative diagnostic and prognostic tools. Researchers employ the ADNI dataset for various tasks, such as disease prediction, progression tracking, and the evaluation of treatment interventions, ultimately contributing to advancements in Alzheimer's disease research and patient care.

## Project Structure

### Files

The project is organized into several files and folders:

- `modules.py`: Contains the implementation of the ViT model components, including patches, patch encoding, and the main classifier.
- `dataset.py`: Contains data loading and preprocessing functions to prepare the image dataset for training.
- `train.py`: Trains the ViT model on the provided dataset and saves the trained model.
- `predict.py`: Uses the trained model to make predictions on single images and visualize the results.

## Vision Transformer (ViT) Model Design

### Introduction

The Vision Transformer (ViT) is a powerful deep learning architecture for image classification tasks. It leverages the self-attention mechanism to capture global and local relationships within an image. This design provides an overview of the model architecture and key components for Alzheimer's disease image classification.

#### Model Architecture

The ViT model consists of the following main components:

1. **Input Layer**: The input to the model is an image with dimensions (image_size, image_size, 3), where 3 represents the RGB channels.

2. **Data Augmentation**: To enhance model generalization, data augmentation techniques, such as random flips, rotations, and zoom, are applied to the input images during training. These augmentations help the model learn from variations in the data.

3. **Patch Extraction**: The input image is divided into non-overlapping patches of size (patch_size x patch_size), effectively dividing the image into a grid of patches. These patches are then linearly embedded into flattened vectors.

4. **Positional Encoding**: A positional encoding is added to the patch embeddings to provide spatial information to the model. It enables the model to understand the relative positions of patches in the image.

5. **Transformer Encoder**: The core of the ViT model is a stack of transformer encoder layers. Each encoder layer consists of multi-head self-attention and feedforward sub-layers. These sub-layers enable the model to capture global and local relationships within the image. Multiple encoder layers are stacked to learn hierarchical features.

6. **Classification Head**: The output of the encoder stack is passed to a classification head. The classification head can vary depending on the task. For Alzheimer's disease classification, a dense feedforward layer followed by a sigmoid activation function is commonly used to predict the probability of the input image belonging to a particular class (AD or CN).

#### Hyperparameters

- `image_size`: The size of the input image (e.g., 128x128).
- `patch_size`: The size of patches extracted from the image (e.g., 16x16).
- `num_heads`: The number of self-attention heads in the transformer encoder.
- `transformer_units`: The number of units in the feedforward layers within the transformer encoder.
- `num_layers`: The number of transformer encoder layers.
- `dropout_rate`: Dropout rate applied to the model for regularization.
- `learning_rate`: The learning rate for training the model.

## Usage
- To train the model, run `train.py` and provide the necessary arguments.
- To make predictions on a single image, run `predict.py` and provide the path to the image file.

## Requirements
- Python 3.7+
- TensorFlow
- Numpy
- OpenCV
- Matplotlib


