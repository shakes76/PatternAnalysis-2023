# COMP 3710 Pattern Analysis Report
**Author**: Gaurika Diwan
**Student ID**: 48240983
**Description**: The Given Project contains the files for COMP3710 project report for the Year 2023.

## Table of Contents

1. [Introduction](#introduction)
2. [About VIT](#about-vit)
3. [Dependencies](#dependencies)
4. [Directory Structure](#directory-structure)
5. [Elaboration on Dataset.py](#elaboration-on-datasetpy)
   - Training Data(#training-data)
   - Testing Data(#testing-data)
   - Dataset Class(#dataset-class)
   - Application(#application)
6. [Elaboration on Modules.py](#elaboration-on-modulespy)
    - Embedder Module(#Embedder-Module)
    - Vision Module (#Vision-Module)
    - Model Usage (#model-usage)
7. [Elaboration on Train.py](#elaboration-on-trainpy)
    - Neural Network 
    - Optimizer (Stochastic Gradient Descent - SGD)
    - Model Training
    - Usage
8. [Elaboration on Predict.py](#elaboration-on-predictpy)
    - Overview 
9. [Changes Made after 20th]
10. [Results Shown](#results-shown)
11. [References](#references)

## Introduction

**Chosen Task**: Classify Alzheimer’s disease (normal and AD) of the ADNI brain data using a visual or transformer set having a minimum accuracy of 0.8 on the test set. (Hard Difficulty)

### About The Project
The project focuses on classifying Alzheimer’s disease using neurological data, including MRI and PET scans. The dataset is divided into a training set and a test set, with preprocessing steps like image resizing and data augmentation.

 Pre-processing data is essential. Possible actions are:

1. Image resizing: Model training demands consistent image sizes.

2. Data augmentation: Enhance the amount of training instances produced by training data using techniques like flipping, revolving, and cropping.

3. Scaling pixel values to a normal range, such as [ 0 ] or [-1, 1], is  normalisation.


### About VIT
The Vision Transformer (VIT) extends the Transformer architecture for computer vision tasks. Key components include patch organization, self-attention, and token embeddings, leading to outstanding performance in image classification, object detection, and segmentation.

### Key components of VIT's operation include:
1.	Instead of being placed in a 2D grid like CNNs, patchwork are organised consecutively like syllables in a sentence, removing the requirement for spatial links.

2.	As a way to capture long-range dependencies, the Transformer encoder determines self-attention between each set of patches by considering a series of patch insertions like a text sequence.

3.	To maintain the exact spot of each patched in the original image, placement insertions are learned and joined to the patch embeddings.
4.	The class token embedding, that conveys a global awareness of the image contents, uses for categorization after a lot Transformer encoder layers.

5.	VIT's general modelling skills enabled it show exceptional results even with restricted datasets and transfer properly to new sectors.
(image)

The principal uses for Vision Transformers are:


1.	Image Classification: On multiple image classification benchmarks, such as ImageNet, VIT has produced breakthrough findings.

2.	Object Detection: By introducing bounding box coordinates to the image patches, VIT can be used for object detection jobs.


3.	Image Segmentation: By treating each pixel as a patch, VIT models have been used for semantic and instance segmentation.

4.	Image Generation: To generate new images, decoders use features derived from images by VIT encoders.


5.	Video Understanding: VIT allows for spatiotemporal links in videos for objectives like action recognition by comprehending frames as patches.

### Dependencies

Before running the code,  installed the required Python libraries:

- PyTorch
- torchvision
- PIL (Pillow)
- Multi-Layer Perceptrons (MLPs)

### Directory Structure
 
(image)


### Elaboration on Dataset.py

The `Dataset.py` module utilizes the ADNI brain dataset, containing two main directories (`train` and `test`) with subdirectories (`AD` and `NC`). It labels images as 1 (AD) or 0 (NC) based on their location. The `CustomDataset` class facilitates image loading, transforms, and batch creation.These subdirectories house contains brain image of patients, with each patient having a total of 20 brain image slices. 

## Training Data:

AD (Alzheimer's Disease): This category contains 10,400 images.

NC (Normal Control): This category contains 11,120 images.

AD                                             NC
                     
## Testing Data:

AD (Alzheimer's Disease): This category contains 4,460 images.

NC (Normal Control): This category contains 4,540 images.

AD                                                       NC
                     
## Image labelling:
 If the image is found in the “AD” folder, then it will be labelled as 1, otherwise 0.

## Dataset Class 

The CustomDataset class takes images  from PyTorch's Dataset class. It takes the data directory and selected transforms as arguments.
Each subdirectory represents a class.

The required Dataset methods 'len' and 'getitem' are implemented to return the number of samples and fetch a sample respectively.


### Application

1. Passed the data directory to begin setting up the dataset.

2. Used training picture changes.

3. For training, built a DataLoader to batch the samples.

4. Repeat the dataloader repeatedly in a training loop.

## Elaboration on Modules.py

The `Modules.py` file defines modules for image classification. The `Embedder` module extracts image patches, while the `VisionModel` combines embeddings, transformers, and an MLP head. It contains crucial parameters like the number of classes and model dimensions.

### Embedder Module

The Embedder module is in charge of transforming incoming photos into a collection of image patches so that the model can be processed further. Parameters are defined as follows:

1. patch_size: An image patch's size.
2. dim: The embedding's dimension.
3. encoder: A sequential module that uses convolutional layers to extract patches and then batches normalisation.
4. forward: A forward pass technique that flattens embeddings after applying the encoding process to input image data.

### VisionModel Module

The Embedder, transformer layers, and an MLP head for categorization are all united in the VisionModel module, forming the basis of the ViT architecture. Amongst its crucial elements are:

1. num_classes: the number of image classification output classes.
2. image_size: The provided image's dimensions (height, width).
3. patch_size: An image patch's size.
4. num_patches: The sum of all the patches within the picture.
5. dim: The embedding's dimension.
6. depth: The quantity of layers in a transformer encoder.
7. num_heads: The transformer's total number of attention heads.
8. embedder: An image patch processing instance of the Embedder module.
9. token: An embedding parameter for tokens.
10. positions: A positional embedding parameter.
11. transformer: An embedding processing transformer layer.
12. mlp_head: A perceptron head with several layers.

## Model Usage

1. Set the hyperparameters (`num_classes`, `image_size`, `patch_size`, `num_patches`, `dim`, `depth`, `num_heads`, `mlp_dim`) according to your requirements.
2. Create an instance of the `VisionModel` by providing these hyperparameters.
3. Use the model to make predictions or include it in your custom image classification pipeline.


## Elaboration on Train.py

The `Train.py` script trains a  model to classify Alzheimer's disease. It defines the model, loss function, and optimizer, conducts the training loop, and uses the SGD optimizer. Training data is processed over 10 epochs.
This section goes into more detail about the model architecture. In this case, it talks about the Deeper Model that consists of three fully connected layers with 'ReLU' activations.

###  Neural Network 

The neural network defined in your provided code is a simple feedforward neural network for regression tasks. It consists of three fully connected (dense) layers with Rectified Linear Unit (ReLU) activations. Here's an elaboration on the neural network:

1. Input Layer:

It accepts one-dimensional input data, which is a single numerical feature.
In the code, this layer corresponds to the feature data  to use for regression.

2. Fully Connected Layer 1:

The first hidden layer is a fully connected layer, also known as a dense layer.
It has 10 neurons, meaning it will produce a 10-dimensional output.
The ReLU activation function is applied to each neuron's output.
ReLU (Rectified Linear Unit) is an activation function that introduces non-linearity by outputting zero for negative values and passing positive values unchanged.

3. Fully Connected Layer 2:

The second hidden layer is another fully connected layer.
It has 10 input features (coming from the 10 neurons of the previous layer) and 5 output features.
Like the previous layer, ReLU activation is applied to the output of each neuron.

4. Fully Connected Layer 3 (Output Layer):

The final layer of the neural network consists of a single neuron.
It takes the 5-dimensional input from the previous layer and produces a single output.
This output is the predicted value of the regression task.
The ReLU activation is also applied to the output of this neuron.

### Optimizer (Stochastic Gradient Descent - SGD)

The code uses the Stochastic Gradient Descent (SGD) optimizer to train the neural network. Here's an explanation of the optimizer's role:

Initialization:

optimizer = 'optim.SGD(model.parameters()', lr=0.01): This line initializes the SGD optimizer.(learning rate changed for the updated graph plots)

model.parameters(): It passes the model's parameters to the optimizer. These parameters include the weights and biases of the neural network layers.

Learning Rate (lr=0.01):  A learning rate of 0.01 means that in each training iteration, the optimizer will make relatively small updates to the model's parameters. The learning rate is a crucial hyperparameter that can affect training stability and convergence.

### Model Training

In this section, the training process of the model is explained:

The optimizer (SGD) and loss function (MSE) are defined.

The process of model training is briefly described, indicating that training and test loss are monitored and saved.(Updated Code For more accurate values)

The goal of the model is to minimize the mean squared error (MSE) loss when making predictions. The results are saved to a text file, and both training and test loss and accuracy are visualized using Matplotlib, with the plots saved as an image file for further analysis and evaluation of the model's performance.

### Usage

This part outlines how users can use this project:

1. Set the hyperparameters: Explains that users can configure key settings, such as the learning rate (lr), batch size, and the number of training epochs.

2. Create an instance of the Net class: Provides a high-level overview of how to create the model instance.

3. Define the optimizer and loss function: Informs users about the necessary steps to configure the training process.

4. Train the model: Describes the training process using the user's dataset and how to monitor loss and accuracy.

5. Visualize the results: Indicates that users can visualize results using Matplotlib and save the plots as image files.





## Elaboration on Predict.py

The `Predict.py` script loads a pre-trained PyTorch model, makes predictions, calculates accuracy and confidence, and generates bar charts. Confidence scores are visualized for training and testing data, and the results are saved as an image.

## Overview

1. Model Loading: 

The script loads a pre-trained ResNet-18 model from PyTorch's model hub for image classification. The model has been pre-trained on a large dataset and can effectively classify images.

2. Image Preprocessing: 

Image preprocessing transformations are set up. This includes resizing the image to a specific size, converting it to a tensor, and normalizing it according to pre-defined mean and standard deviation values.

3. Image Classification: 

The script loads and preprocesses the input image from the specified image_path. The pre-trained model is used to predict the class of the image ('AD' or 'NC'). A confidence plot is created, showing the model's confidence in its prediction.

4. Visualization: 

The input image is visualized and saved as 'input_image.png'.
A confidence plot, indicating the confidence of the model's prediction, is created and saved as 'training_testing_confidence_scores.png'.
The predicted class ('AD' or 'NC') is displayed.

5. Prediction Result:

The predicted class ('AD' or 'NC') is printed in the console.

## Results Shown

The updated accuracy achieved on the testing data is 76%, while the training accuracy over 10 epochs is 88%. Training and testing loss exhibit a decreasing trend over epochs. The accuracy graph shows improvement in training accuracy, and in testing accuracy by the 10th epoch.

## Changes Made after 20th Oct

1. The code uses the Stochastic Gradient Descent (SGD) optimizer to train the neural network, as with experiencing memory issues with Adam, switching to SGD  alleviate those problems. In my model, SGD  lead to more consistent and interpretable training dynamics.

2. FeedForward Network Used, MLPs used in my model has shown well-suited values for regression tasks where the model predict continuous numerical values. (Accuracy and loss)

3. Learning rate changed for the updated graph plots which is 0.01.



## References
1. [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/)
2. [PyTorch Vision - Vision Transformer](https://pytorch.org/vision/main/models/vision_transformer.html)
3. [The Vision Transformer Model](https://machinelearningmastery.com/the-vision-transformer-model/)
