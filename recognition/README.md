# Visual Transformer (ViT) for classifying Alzheimer's Disease

## OVERVIEW:
This project is dedicated to creating a machine learning model for the classification of Alzheimer's disease (AD) and normal brain scans, employing advanced Visual or Perceiver Transformer models. The primary objective is to achieve a minimum accuracy of 0.8 on the test dataset
n 2020, the groundbreaking paper titled "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" demonstrated that traditional Convolutional Neural Networks could be surpassed by Vision Transformers. These Vision Transformers proved capable of delivering outstanding results when compared to state-of-the-art convolutional networks, all while demanding fewer computational resources for training.The adoption of Vision Transformers in this project is driven by the potential to harness their efficiency and accuracy in medical image classification, ultimately contributing to the advancement of Alzheimer's disease diagnosis and enhancing the healthcare landscape.

![visual transformer](https://github.com/saakshigupta2002/PatternAnalysis-2023/assets/62831255/579168d1-8dbe-4177-a549-52b8a930319c)

##MODEL ARCHITECTURE:
The Visual Transformer (ViT) is an influential neural network architecture for computer vision, adapting the Transformer model from natural language processing to process images. ViT divides images into patches, transforms them into vectors, and employs a multi-head self-attention mechanism to capture complex spatial relationships. It uses stacked layers of self-attention and feedforward networks to extract features and make predictions. ViT excels in tasks like image classification and object detection, thanks to its ability to handle global and local information. However, training ViT models usually requires pre-training on large image datasets due to its high parameter count, yet it has significantly impacted the field of computer vision.

The Vision Transformer model consists of the following steps:
1.Split an image into fixed-size patches
2.Linearly embed each of the patches
3.Prepend [class] token to embedded patches
4.Add positional information to embedded patches
5.Feed the resulting sequence of vectors to a stack of standard Transformer Encoders
6.Extract the [class] token section of the Transformer Encoder output
7.Feed the [class] token vector into the classification head to get the output

##Transformer Encoder: 
The Transformer Encoder is composed of two main layers: Multi-Head Self-Attention and Multi-Layer Perceptron. Before passing patch embeddings through these two layers, we apply Layer Normalization and right after passing embeddings through both layers, we apply Residual Connection.

 ![image](https://github.com/saakshigupta2002/PatternAnalysis-2023/assets/62831255/871fe1ac-dd5c-408a-a6f4-71afb08b3fde)

##Dependencies:
1.Python 3.10.4
2.Tensorflow 2.10.0: An open-source machine learning framework.
3.Tensorflow Addons 0.18.0: An extension library for TensorFlow, providing additional functionalities.
4.Matplotlib 3.5.2: A data visualization library used for creating plots and charts in Python.
5. Keras 2.0.8: A high-level neural networks API that runs on top of TensorFlow.

##Repository Overview:
parameters.py: Stores hyperparameters for model configuration.
modules.py: Contains the Vision Transformer's fundamental components.
dataset.py: Manages data loading functions.
train.py: Compiles and trains the model with relevant functions.
predict.py: Enables model predictions with its functions.

##Alzheimer's Disease Neuroimaging Initiative(ADNI) Dataset:
The dataset comprises 21,500 grayscale images, each with dimensions 256x240, divided into 21,500 training samples and 9,000 test samples. These images fall into two distinct categories: Alzheimer's Disease (AD) patient brain scans and those representing normal cognitive condition (NC).
1.	Training Set: 21,520 images 
2.	Validation Set: 4500 images 
3.	Testing Set: 4500 images 



 

