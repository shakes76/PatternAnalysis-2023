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
The project is organized into several files and folders:

- `modules.py`: Contains the implementation of the ViT model components, including patches, patch encoding, and the main classifier.
- `dataset.py`: Contains data loading and preprocessing functions to prepare the image dataset for training.
- `train.py`: Trains the ViT model on the provided dataset and saves the trained model.
- `predict.py`: Uses the trained model to make predictions on single images and visualize the results.

## Usage
- To train the model, run `train.py` and provide the necessary arguments.
- To make predictions on a single image, run `predict.py` and provide the path to the image file.

## Requirements
- Python 3.7+
- TensorFlow
- Numpy
- OpenCV
- Matplotlib


