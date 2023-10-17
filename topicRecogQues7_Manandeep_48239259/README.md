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
## How to Use

1. **Training the Model**:
   - Run the `train.py` script to train the Siamese Network. The script includes the following steps:
   - Data loading from the prepared dataset.
   - Siamese network architecture and loss function setup.
   - Training loop with adjustable configurations.

2. **Classifying Test Images**:
   - To classify a test image, use the `classify_test_image` function in `predict.py`. Replace the `test_image_path` variable with the path to your test image, and it will provide a classification result.
    - Loading the trained model.
    - Preprocessing the test images.
    - Classifying the test images as "Has Alzheimer Disease" or "Is Cognitive Normal."


3. **Calculating Accuracy**:
   - Use the `calculate_accuracy` function in `predict.py` to calculate the accuracy of the trained model on a test dataset. You can specify the true labels and the test folder path.
   Make sure to replace the `test_image_path` variable with the path to your test image in the `predict.py` script.

## Classification Process

The classification of test images is accomplished through a multi-step process, as detailed below:

### Feature Extraction

The Siamese Network architecture comprises two identical subnetworks, often referred to as the "Siamese twins." These subnetworks share the same architecture and weights. Each subnetwork takes an input image, which can be either a test image or a reference image, and extracts feature representations from it.

### Pairwise Comparison

To classify a test image, it is compared with reference images from both categories: "Has Alzheimer Disease" and "Is Cognitive Normal." This comparison is conducted in pairs, with the test image compared separately with each reference image.

### Feature Similarity

The feature representations extracted from the test image and each reference image are used to compute a similarity score. A common metric for measuring similarity is the Euclidean distance (L2 distance) between the feature vectors. A smaller distance implies greater similarity, while a larger distance indicates dissimilarity.

### Classification Decision

The Siamese Network employs the computed similarity scores to make a classification decision. The decision is typically reached by comparing the similarity scores of the test image with the reference images. The decision-making process is as follows:

- If the test image is more similar to the reference image from the "Has Alzheimer Disease" category, it is classified as "Alzheimer Disease."
- If the test image is more similar to the reference image from the "Is Cognitive Normal" category, it is classified as "Cognitive Normal."

### Thresholding

In certain scenarios, a threshold may be applied to the similarity scores to make a binary decision. If the similarity score with either reference image surpasses a predefined threshold, the test image is classified accordingly.

### Result

The Siamese Network outputs the final classification of the test image as either "Has Alzheimer Disease" or "Is Cognitive Normal."

This classification process enables the model to effectively categorize test images based on their similarity to reference images from known categories.

## Results

The trained Siamese Network achieved the following results:

- Test Accuracy: >= 84%

Users can refer to the 'Testing and Classification' section for more details on using the model.

## Dependencies

To run this project, you need the following software, libraries, and modules:

- Python 3.x
- PyTorch 1.x
- torchvision
- Pillow (PIL)
- Matplotlib (for data visualization)
- torch.nn.functional (F) - PyTorch module for various neural network functions.
- torch.autograd.Variable - PyTorch module for automatic differentiation.
- torch.nn (nn) - PyTorch's neural network module.
- os - Python's built-in module for interacting with the operating system.

You can install the required dependencies using `pip` if they are not already installed.

## References

Here are some of the references and resources that were instrumental in the development of this project:

1. **Dataset**  
   Dataset used for training and testing. [Download Link](https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI/download)

2. **Siamese Neural Networks for One-shot Image Recognition**  
   A reference to the Siamese Network concept. [Read more](https://link.springer.com/protocol/10.1007/978-1-0716-0826-5_3)

3. **Exploring Simple Siamese Representation Learning**  
   Research paper discussing Siamese Representation Learning. [Read more](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.html)

4. **Contrastive Loss for Siamese Networks**  
   Understanding the contrastive loss used in Siamese Networks. [Read more](https://link.springer.com/chapter/10.1007/978-3-319-48881-3_56)

5. **Contrastive Loss Function for Siamese Network**  
    Stack Overflow discussion on applying contrastive loss to Siamese Networks. [Read more](https://stackoverflow.com/questions/54091571/contrastive-loss-function-apply-on-siamese-network-and-something-wrong-with-opti)
