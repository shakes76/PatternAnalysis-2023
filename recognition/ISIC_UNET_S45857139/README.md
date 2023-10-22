# Segmentation of the ISIC 2018 Dataset with the Improved UNET
**Developed by Thomas Witherspoon for COMP3710 (UQ)**

## Introduction
A typical UNET algorithm is characterized by an encoding path that captures the context of an input image, and a decoding path that restores the spatial details. Skip connections between the two paths enhance performance.

My improved UNET implementation, inspired by [this paper](https://arxiv.org/abs/1802.10508v1), uses residual blocks instead of standard convolutions. These residual blocks consist of two convolution layers followed by instance normalization and a dropout layer set to 0.3. The resultant output is added to the residual to create the final output. Furthermore, the improved UNet incorporates a localization module for the decoding path. This module features a 3x3 convolution succeeded by a 1x1 convolution, effectively reducing the channel count by half.

## Task Overview
The goal was to segment the ISIC dataset using the improved UNet, ensuring all labels attain a minimum dice similarity coefficient of 0.8 on the test set. The dataset encompasses images of skin lesions and their corresponding segmented mask images. The model's purpose is to accurately predict the segmented mask for an unseen skin lesion image.

In terms of dataset allocation, 80% is designated for training, while the remaining 20% is used for validation. There's no separate test set. A more substantial training dataset promotes superior model generalization and diminishes overfitting.

## Python Script Functions
1. `modules.py` - Contains the source code for model components. Every component should be implemented as either a class or a function.
2. `dataset.py` - Houses the data loader for loading and preprocessing the dataset.
3. `train.py` - Features the source code for training, validating, testing, and saving the model. Ensure to import the model from `modules.py` and the data loader from `dataset.py`. Also, plot the losses and metrics during the training phase.
4. `predict.py` - Demonstrates the trained model's usage. Be sure to print results and/or provide relevant visualizations where needed.

## Dependencies/Libraries:
- PyTorch
- Python
- PIL
- os

## Training Details
The model underwent training on the Google Colab V100 GPU, spanning 30 epochs. In order to replicate the training, run the predict.py script with the default parameters (n_epochs=30). To run on google colab, I commented out the lines that import functions from other python scripts, and instead, simply pasted my code in the notebook and ran in order. Although, instead, you can just run the predict script with all other files in the same directory.

The results are as follows:

**Sample Predictions:**:

**Sample 1 Input:**
**Sample 1 Mask:**
**Sample 1 Output:**
**Sample 2 Input:**
**Sample 2 Mask:**
**Sample 2 Output:**



[![Sample Predictions](/images/sample1.png)](https://github.com/lombo9/PatternAnalysis-2023/blob/topic-recognition/images/sample1.png) [![Sample Predictions](/images/sample2.png)](https://github.com/lombo9/PatternAnalysis-2023/blob/topic-recognition/images/sample2.png)


**Metrics**: Loss per Epoch and Average Dice Coefficient per Epoch: [![Sample Predictions](/images/results.png)](https://github.com/lombo9/PatternAnalysis-2023/blob/topic-recognition/images/results.png)
