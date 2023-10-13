# Siamese Network Classifier for Alzheimer's Disease Detection
## Description
This repository contains a Siamese Network-based classifier designed to identify Alzheimer's disease (AD) in brain data from the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset. The goal is to achieve an accuracy of approximately 0.8 on the test set by classifying brain images as either normal or AD.

## How It Works
The Siamese Network architecture utilizes a pair of neural networks that share the same weights. It takes two input images and computes their feature vectors. The network then measures the similarity between these vectors, allowing us to determine whether the images belong to the same class or not. In our case, this network is trained to determine if two brain images are from AD patients or healthy individuals. The figure below illustrates the Siamese Network architecture:

## Siamese Network Architecture

Dependencies
To run this code, you'll need the following dependencies:

Python (version X.X)
* TensorFlow (version X.X)
* Keras (version X.X)
* NumPy (version X.X)
* Matplotlib (version X.X)
To ensure reproducibility of results, it's recommended to create a virtual environment and specify the exact versions of the dependencies used in a requirements.txt file.

Example Inputs, Outputs, and Plots
Inputs
The inputs to the Siamese Network classifier are pairs of brain images, one from an AD patient and one from a healthy individual. These image pairs are provided in the ADNI dataset.

Outputs
The output of the classifier is a binary classification result, indicating whether the input image pair is classified as normal or AD.

Plots
The following plots can be generated:

Training Loss and Accuracy: A plot of training loss and accuracy over epochs to visualize the training progress.
Confusion Matrix: A confusion matrix to evaluate the classifier's performance on the test set.
Pre-processing
Pre-processing steps for the ADNI dataset may include resizing, normalization, and data augmentation, depending on the specific network architecture and requirements. References to pre-processing techniques applied can be found in the code and associated documentation.

Data Splitting
The dataset is divided into three sets:

Training Set: Used to train the Siamese Network.
Validation Set: Employed to fine-tune the model and prevent overfitting.
Test Set: Used to evaluate the model's performance and achieve the target accuracy of approximately 0.8.
The specific data splitting and strategies for handling class imbalance are detailed in the code and associated documentation.

For more detailed information, code implementation, and instructions on running the classifier, please refer to the accompanying Jupyter Notebook or Python script.
