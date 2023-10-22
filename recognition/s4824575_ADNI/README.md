<h3 align="center">Siamese Neural Network for AD/NC Classification </h3>


---

<p align="center"> Name: <b>Raghavendra Singh Gulia</b><br>UQ StudentID: <b>s4824575</b>
    <br> 
</p>

## 📝 Table of Contents

- [Introduction](#-introduction-)
- [Code Overview](#-code-overview-)
- [Results](#-results-)
- [Dependencies](#-dependencies-)
- [Conclusion](#-conclusion-)
- [References](#-references-)


## 🧐 Introduction <a name = "introduction"></a>


The Vision Transformer is a powerful deep learning architecture for computer vision tasks. In this project, we develop a Vision Transformer model to classify brain MRI images from the ADNI dataset into two classes: Alzheimer's Disease (AD) and Normal Control (NC). The Vision Transformer is trained to distinguish between these two classes using self-attention mechanisms.
The project is divided into three main components: 
1) Dataset preparation: Loading and preprocessing the ADNI brain MRI images, 
2) Model architecture: Defining the Vision Transformer architecture based on the ViT paper [8,9], 
3) Training and Prediction: Training the model on preprocessed data and evaluating performance on the held-out test set with the target of achieving a minimum accuracy of 0.8.
The Vision Transformer takes advantage of self-attention to capture global relationships between image patches, without relying on local connections like convolutional networks. This allows it to efficiently model long-range dependencies in the brain MRI data to distinguish between AD and normal classes. We implement the model in TensorFlow and evaluate its ability to classify Alzheimer's disease from brain images with the goal of supporting early disease detection.

![Vision Transformer { width="800" height="600" style="display: block; margin: 0 auto" }](/Users/raghavendrasinghgulia/PatternAnalysis-2023/recognition/s4824575_ADNI/VisionTransformer.png)

## 👨🏻‍💻 Code Overview <a name = "code_overview"></a>

<p> This project aims to develop a machine learning model to classify MRI brain images as either Alzheimer's disease (AD) or normal control (NC) using data from the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset. A Vision Transformer architecture is implemented and trained on preprocessed MRI images to distinguish between the two classes<p>
<ol>

<b><li>Dataset Preparation</li></b>
<p>The ADNI dataset contains structural MRI brain scans from multiple sites for AD patients and healthy controls. In the dataset preparation phase, images are loaded from directories organized by class label. The dataset is then split into training, validation, and test subsets for model development and evaluation. Preprocessing steps like resizing, normalization, and grayscale conversion ensure consistency across images.
</p>
<b><li>Model Architecture</li></b>
<p>A Vision Transformer model is defined using the TensorFlow Hub module API. It consists of an embedding layer to generate patch embeddings from input images, followed by an encoder stack of multi-head self-attention and feedforward blocks to learn contextual relationships. Classification is performed via dense layers at the end.
</p>
<b><li>Training </li></b>
<p>The model is compiled with an optimizer, loss function and metrics. It is trained on the preprocessed training subset for 10 epochs with validation monitoring to prevent overfitting. The model weights yielding the best validation accuracy are saved for inference. Hyperparameters like learning rate, batch size are tuned.

-	<b>Data Loading:</b> The training data is loaded using the load_data function, is created.
-	<b>Device Selection:</b> The code checks for GPU availability and moves the model to the GPU for accelerated training.
-	<b>Training Loop:</b> The model undergoes training for a predetermined number of epochs. During training, the Vision Transformer learns to distinguish between AD and NC images.
-	<b>Model Saving:</b> After training, the model is saved to a file as model.h5 (not pushed to the repo since the file was big), allowing for future use.
-	<b>Loss Plotting:</b> A plot of training loss is generated to visualize the training progress.
</p>
<b><li>Making Predictions</li></b>
<p>The prediction phase is executed using the <b>predict.py</b> script. The main components of this phase include:
raining curves are plotted to visualize the learning process. The trained model is loaded on test data and the final test accuracy is logged. The project outputs a classification model that meets the performance criteria on unseen MRI examples.
</p>
</ol>

