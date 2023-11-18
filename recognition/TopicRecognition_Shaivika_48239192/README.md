# Alzheimer Image Classification with Vision Transformer (ViT) 

## Overview
This project is an implementation of a Vision Transformer (ViT) for image classification. The code includes components for loading, preprocessing, and training a ViT model on ADNI dataset. Additionally, it allows you to make predictions and provides metrics for model evaluation.

## Table of Contents
- [ADNI Dataset](#adni-dataset)
- [Project Structure](#project-structure)
  - [Vision](#vision)
  - [Files](#files)
    - [Transformer (ViT) Model Design Introduction](#transformer-vit-model-design-introduction)
    - [Model Architecture](#model-architecture)
    - [Hyperparameters](#hyperparameters)
    - [Training](#training)
    - [Results](#results)
      - [Resizing image](#resizing-image)
      - [Dividing images into Patches](#dividing-images-into-patches)
      - [Model loss graph](#model-loss-graph)
      - [Model accuracy graph](#model-accuracy-graph)
      - [Heatmap](#heatmap)
      - [Accuracy](#accuracy)
      - [Actual vs Predicted image analysis](#actual-vs-predicted-image-analysis)
  - [Data Preprocessing](#data-preprocessing)
- [Data](#data)
- [Validation Evidence](#validation-evidence)
- [Discussion](#discussion)
- [Usage](#usage)
- [Requirements](#requirements)
- [References](#references)

## ADNI Dataset
The Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset is a valuable and widely used collection of medical imaging and clinical data aimed at advancing our understanding of Alzheimer's disease. Comprising a comprehensive range of neuroimaging modalities, including MRI and PET scans, as well as clinical and cognitive assessments, the ADNI dataset has played a pivotal role in enhancing research related to neurodegenerative diseases. This dataset not only facilitates the identification of biomarkers associated with Alzheimer's disease but also promotes the development of innovative diagnostic and prognostic tools. When we first open the dataset, we are able to see 2 sets of different folders named ‘ AD’ and ‘CN’. Upon further exploration we see the images in it.

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
- `image_size`: The size of the input image .
- `patch_size`: The size of patches extracted from the image .
- `num_heads`: The number of self-attention heads in the transformer encoder.
- `transformer_units`: The number of units in the feedforward layers within the transformer encoder.
- `num_layers`: The number of transformer encoder layers.
- `dropout_rate`: Dropout rate applied to the model for regularization.
- `learning_rate`: The learning rate for training the model.

#### Training
- Loss Function: Categorical Cross-Entropy (for multi-class classification) or Binary Cross-Entropy (for binary classification, such as AD vs. CN).
- Optimizer: Adam optimizer.
- Training Data: The ADNI dataset with labeled images.
- Evaluation Metric: Accuracy

### Results
The application of the Vision Transformer (ViT) model for Alzheimer's disease image classification has yielded promising results. The trained ViT model demonstrates its effectiveness in accurately distinguishing between Alzheimer's disease (AD) and normal control (CN) subjects based on medical images.

Key results and highlights:

1. **High Accuracy**: The ViT model achieves a high accuracy rate on the ADNI dataset, surpassing the minimum accuracy threshold of 0.8 (80%) as specified in the project requirements.

2. **Confusion Matrix**: The evaluation of the model's performance includes a confusion matrix that provides insights into true positives, true negatives, false positives, and false negatives. This matrix aids in assessing the model's classification effectiveness.

3. **Individual Image Predictions**: The `predict.py` script allows for the classification of individual images. This feature is useful for making predictions on new, unseen images to identify potential cases of Alzheimer's disease.

4. **Visual Representation**: The script not only provides numerical results but also offers a visual representation of model predictions. A set of random test images, along with their actual and predicted labels, allows for a qualitative assessment of the model's performance.

#### Resizing image
![Resized image loading..][recognition/TopicRecognition_Shaivika_48239192/output_images/image.png]

#### Dividing images into Patches
![Patched image loading..][recognition/TopicRecognition_Shaivika_48239192/output_images/patches.png]

#### Model loss graph
![Model loss graph image loading..][recognition/TopicRecognition_Shaivika_48239192/output_images/model_loss.png]

#### Model accuracy graph
![Model accuracy graph image loading..][recognition/TopicRecognition_Shaivika_48239192/output_images/model_accuracy.png]

#### Heatmap
![Heatmap image loading..][recognition/TopicRecognition_Shaivika_48239192/output_images/heatmap.png]

#### Accuracy
![Accuracy image loading..][recognition/TopicRecognition_Shaivika_48239192/output_images/accuracy.png]

#### Actual vs Predicted Analysis
![Actual vs Predicted image loading..][recognition/TopicRecognition_Shaivika_48239192/output_images/actual_vs_prediction.png]

### Data Preprocessing
The data preprocessing phase involves the handling of the ADNI dataset, which comprises tasks such as image resizing and normalization. Within the normalization process, mean and variance values are computed individually for each image. 

### Validation Evidence
Validation of the model's performance is a critical aspect of this project. Here's how validation evidence is obtained:

1. **Data Splitting**: The ADNI dataset is split into training and validation sets. A common split ratio is used, e.g., 80% for training and 20% for validation.

2. **Model Training**: The ViT model is trained on the training set with specific hyperparameters, such as learning rate, batch size, and number of transformer layers.

3. **Model Evaluation**: The trained model is evaluated on the testing set using relevant evaluation metrics, such as accuracy.

4. **Confusion Matrix**: A confusion matrix is generated to assess the model's classification performance, providing insights into true positives, true negatives, false positives, and false negatives.

5. **Accuracy Recording**: The test accuracy is recorded and saved in a png file for reference.

### Discussion
The validation results indicate that the ViT model performs well in distinguishing between Alzheimer's disease (AD) and normal control (CN) subjects. The model demonstrates a high accuracy on the ADNI dataset, surpassing the specified accuracy threshold. The confusion matrix and individual image predictions provide a detailed understanding of the model's classification effectiveness.

The accuracy of the model is [94]% as recorded in the png file 'accuracy.png'.

The visual representation of sample images and their actual/predicted labels demonstrates the model's capability to make accurate predictions. This performance highlights the significance of the ViT model in the early diagnosis and classification of Alzheimer's disease based on medical images.

## Usage
To make predictions using the trained Vision Transformer (ViT) model, you can utilize the provided `predict.py` script. This script allows you to assess the model's performance on both individual images and the entire test dataset.

**Individual Image Prediction**:

1. Load the trained ViT model, which has been fine-tuned for Alzheimer's disease classification.
2. Preprocess the image you wish to classify by resizing it to the specified `image_size` and normalizing pixel values.
3. Pass the preprocessed image through the model to obtain predictions.
4. For binary classification (e.g., AD vs. CN), you can set a threshold (e.g., 0.5) to determine the class based on the probability output.

**Test Dataset Evaluation**:

1. The script loads the test dataset and makes predictions on all the test images.
2. It generates a confusion matrix to assess the model's classification performance.
3. The confusion matrix provides insights into the true positives, true negatives, false positives, and false negatives.

Additionally, the script showcases a set of random test images along with their actual and predicted labels, giving you a visual representation of the model's classification results.

You can further customize and extend the `predict.py` script to suit your specific needs for image classification and evaluation. Be sure to adjust any file paths or parameters as necessary to work with your dataset.

## Requirements
- Python 3.7+
- TensorFlow
- Numpy
- OpenCV
- Matplotlib

## References 
- https://adni.loni.usc.edu/
- https://www.kaggle.com/datasets/katalniraj/adni-extracted-axial
- https://medium.com/data-and-beyond/vision-transformers-vit-a-very-basic-introduction-6cd29a7e56f3
- https://paperswithcode.com/method/vision-transformer
- https://viso.ai/deep-learning/vision-transformer-vit/#:~:text=Vision%20Transformers%20(ViT)%20is%20an  and%20a%20feed%2Dforward%20layer.
- https://towardsdatascience.com/understand-and-implement-vision-transformer-with-tensorflow-2-0-f5435769093
- https://machinelearningmastery.com/the-vision-transformer-model/