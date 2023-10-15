# Classifying Alzheimer's Disease Using a Visual Transformer
The aim of this project is to classify Alzheimer's disease of the ADNI brain dataset using a visual transformer (ViT). The ADNI brain dataset is composed of two classes - AD meaning Alzheimer's disease, and NC meaning Normal Cognitive. The goal is to perform classification with a minimum accuracy on the test dataset of 0.8.

## Vision Transformer (ViT)
Vision Transformers, often referred to as ViTs, signify a groundbreaking departure from the traditional Convolutional Neural Networks (CNNs) in the realm of computer vision. These models are primarily characterized by their utilization of the Transformer architecture, originally designed for natural language processing but adapted for visual data processing.

![alt text](images/Vision_Transformer.gif)

In a ViT the first step involves dividing an input image into non-overlapping patches of fixed size, typically 16 by 16 pixels. These patches are then linearly embedded into low-dimensional vectors treated as a sequence of tokens. After these patches are linearly embedded a class token is prepended to the sequence. The class token is a learnable embedding focused on representing the entire image, and is hence crucial in classification. All embedded tokens are then added with a positional embedding which is generally just a random tensor representing the location of the embedding on the image.

This sequence of embedded vectors is then processed through a transformer encoder. Within the encoder are multiple feed-forward layers and self-attention layers. This self-attention mechanism is crucial as it allows the model to learn the long-range depedencies between patches. The output of the encoder is fed into a MLP which classifies the image.

## ADNI Dataset
It was mentioned there were two classes in the ADNI dataset, Alzheimer's disease and Normal Cognitive. Below there are examples for both classes (AD top, NC bottom)

![alt text](images/AD_Example.jpeg)

![alt text](images/NC_Example.jpeg)


## Pre-processing
Here talk about pre-processing like image cropping, normalisation, training/test/validation splits

## Training
Here talk about training show plots, different hyperparameter results and that

## Final Model Description

## Test Dataset Accuracy
