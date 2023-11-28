# Perceiver Transformer for Alzheimer's Disease

**Author:** Atharva Gupta (Student ID: 48195609)

## Introduction

The core objective of this project is to discern Alzheimer's Disease (AD) from normal cognitive function (NC) using brain scan images, and to achieve this, we have harnessed the power of the Perceiver transformer. This innovative approach leverages transformer technology for image classification, offering a model that operates with fewer preconceived notions about the underlying data structure. By doing so, it reduces the risk of introducing biases. Unlike traditional convolutional methods that rely on spatial relationships, the Perceiver relies on a latent array to optimize its attention operations. This project represents a departure from conventional techniques, exploring a more adaptable and unbiased approach to Alzheimer's Disease classification.

## Changes to Plan: Shifting from Visual Transformers to Perceiver

Initially, the project's approach centered around the use of Visual Transformers. Visual Transformers extend the original Transformer architecture to handle 2D images, making them suitable for various computer vision tasks, including image classification.

### Visual Transformer Architecture

In the ViT approach, 2D image inputs are first divided into smaller patches. These patches are then transformed into 1D vectors through linear projections, and learnable class embeddings are introduced. To maintain the spatial order of image patches, positional encodings are included. These processed patches, now embedded with class and positional information, are subsequently input into standard Transformer encoders, where interrelationships between patches in the image data are captured. Finally, a Multi-Layer Perceptron (MLP), a neural network that identifies complex relationships, is employed for classification tasks.

However, during the project's execution, several challenges and difficulties arose while implementing the Visual Transformer approach. These difficulties included issues with model convergence, computational resource constraints, and complexities in handling positional information within the framework. As a result, a decision was made to pivot from the Visual Transformer approach to the Perceiver model.

## Shifting to Perceiver

The Perceiver model offers a different perspective by using latent arrays to manage attention operations, ultimately reducing computational complexity. The project transitioned to the Perceiver architecture to address these challenges and explore its potential benefits in the context of Alzheimer's Disease classification using brain scan images. This shift aims to leverage the Perceiver's ability to handle images while making fewer assumptions about data structure, ultimately reducing potential biases and computational demands.

The decision to shift to the Perceiver model reflects the project's adaptability and commitment to exploring the most suitable techniques for the task at hand. While Visual Transformers have proven effective in various computer vision applications, the challenges encountered in this specific project prompted the exploration of alternative approaches, aligning with the project's objective of achieving accurate Alzheimer's Disease classification.

### Perceiver Architecture

This section provides an essential insight into the Perceiver's architecture, elucidating its distinctive ability to grasp intricate data patterns. Its core components, including cross-attention mechanisms, self-attention layers, and latent transformers, work collaboratively to interpret complex image data and make highly accurate classifications.

The Perceiver's adaptability, efficiency, and potential to reduce biases make it a compelling model in the field of medical diagnostics. This introduction offers a glimpse into the Perceiver's architecture, setting the stage for a deeper understanding of its inner workings and the groundbreaking results it can achieve.
The Perceiver's architecture incorporates positional encodings, which facilitate the model's comprehension of the spatial connections among pixels. Although the original research paper presented two techniques for positional encodings (Fourier-based and learned encoding), this project embraces the latter strategy.

Perceiver Architecture:
![Perceiver_transformer](https://github.com/atharvagupta2003/PatternAnalysis-2023/assets/140630788/626c756a-1ce6-4cf4-a7e6-2280e285b24a)


## Repository Overview

- **parameter.py**: Contains the hyperparameters used for specifying data loading and model settings.
- **modules.py**: Contains the components of the Perceiver.
- **datasplit.py**: Contains the function used for loading the data.
- **train.py**: Contains the functions for compiling and training the model.
- **test.py**: Contains the functions for predicting on the trained model.
- **assests**: Contains plotted assets

## Usage

To train the model, run the train.py script, which contains various constants for configuring the Perceiver. These include latent dimensions (N and D), Perceiver depth, batch size, learning rate, and the number of epochs. The trained model will be saved in the path specified by MODEL_PATH, which can be adjusted as needed. NUM_LAYERS must be a multiple of 4.
To make predictions on the testing data and evaluate accuracy, use the test.py script.

## Results

The model training process involved the utilization of Cross-Entropy Loss in conjunction with the Adam optimizer. To accommodate memory constraints, a slightly scaled-down model was employed compared to the one presented in the research paper. This adapted model incorporated a single cross-attend cycle and a latent transformer with a depth of 5, whereas the paper's model used eight cross-attend cycles with latent transformers of depth 8 for ImageNet classification.

Operating with a batch size of 5, a learning rate of 0.005, and 10 epochs, the following outcomes were observed:

- Loss Curve:
  ![loss curve](https://github.com/atharvagupta2003/PatternAnalysis-2023/assets/140630788/001dd37f-be24-483b-9090-15089c74a225)

- Training Curve:
  ![training curve](https://github.com/atharvagupta2003/PatternAnalysis-2023/assets/140630788/70286905-8b99-46e5-8812-c6cece3d6a2a)

The model achieved a test set accuracy of 53%, which is akin to random guessing in the context of binary classification.

The training process encountered several challenges. Despite successfully training the model, the loss function exhibited minimal reduction, indicating that the model's performance did not substantially improve with additional epochs. Attempts were made to fine-tune the model parameters, but due to limited laptop memory, increasing one parameter led to the reduction of others.

It is anticipated that conducting more extensive training on a more powerful computing platform could help address these challenges. The Perceiver model is intricate, and even this streamlined version boasts nearly 8 million parameters. Additionally, the model must acquire an understanding of positional encodings, which constitute the model's initial layer. Achieving this understanding may require extended training and refinement. Consequently, a longer training duration is likely to enhance the loss function and, consequently, improve training accuracy.

## Structure

![structure](https://github.com/atharvagupta2003/PatternAnalysis-2023/assets/140630788/fe2fc5f8-091f-4c24-8349-a5ee79bbac1e)

The code for data loading is contained in **datasplit.py**. The `train_loader` and `test_loader` functions accept a directory path and a batch size as arguments. The directory should contain two subfolders, named 'train' and 'test', each containing 'AD' and 'NC' subfolders with the respective brain scan images. The test/train splits were predefined in the dataset.

To train the model, run the **train.py** script, which contains various constants for configuring the Perceiver. These include latent dimensions (N and D), Perceiver depth, batch size, learning rate, and the number of epochs. The trained model will be saved in the path specified by `MODEL_PATH`, which can be adjusted as needed. `NUM_LAYERS` must be a multiple of 4.
To make predictions on the testing data and evaluate accuracy, use the **test.py** script. Make sure to create the model with the same parameters used for testing.
