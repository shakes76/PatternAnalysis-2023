![Untitled](assets/explorer_I5UTxEbZvw.png)
<h3 align="center">Siamese Neural Network for AD/NC Classification </h3>


---

<p align="center"> Name: <b>Rachit Chaurasia</b><br>ID: <b>s4823870</b>
    <br> 
</p>

## 📝 Table of Contents

- [Introduction](#🧐-introduction)
- [Code Overview](#👨🏻‍💻-code-overview)
- [Detailed Overall Working](#👽-detailed-overall-working)


## 🧐 Introduction <a name = "introduction"></a>

The Siamese Neural Network is a powerful deep learning architecture used for various tasks, including image classification and similarity measurement. In this project, we have developed a Siamese Network to classify medical images into two classes: Alzheimer's Disease (AD) and Normal Control (NC). The network is trained to distinguish between these two classes, and the project is divided into three main components: dataset preparation, neural network architecture, training, and prediction.

## 👨🏻‍💻 Code Overview <a name = "code_overview"></a>

<ol>
<b><li>Dataset Preparation</li></b>
<p>In the dataset preparation phase (<b>dataset.py</b>), we organize the medical image data into a format suitable for training and evaluating the Siamese Network. The key components include:

-	<b>Dataset Classes:</b> Two custom dataset classes, <b>SiameseDataset</b>, are defined for training and testing datasets.
-	<b>Data Loading:</b> The code loads medical image data from the specified directories. The dataset is organized into training and testing sets for AD and NC images.
-	<b>Image Preprocessing:</b> The loaded images are processed to ensure consistency in terms of size, grayscale conversion, and normalization.
</p>
<b><li>Neural Network Architecture</li></b>
<p>The heart of the project is the Siamese Neural Network (<b>modules.py</b>). The network architecture is structured as follows:

-	<b>CNN Architecture:</b> The network consists of a Convolutional Neural Network (CNN) to process the input images. It includes two convolutional layers with max pooling to extract relevant features.
-	<b>Fully Connected (FC) Layers:</b> The output from the CNN is passed through fully connected layers for classification.
-	<b>Two Outputs:</b> The network produces two output vectors, one for each input image in a pair, which are used for comparison during training.
-	<b>Contrastive Loss:</b> A custom loss function, called Contrastive Loss, is employed to train the network. It encourages the network to minimize the distance between similar image pairs and maximize the distance between dissimilar pairs.
</p>
<b><li>Training the Siamese Network</li></b>
<p>The training phase is performed in the train.py script. The key steps are as follows:

-	<b>Data Loading:</b> The training data is loaded using the load_siamese_data function, and a Siamese Network model is created.
-	<b>Device Selection:</b> The code checks for GPU availability and moves the model to the GPU for accelerated training.
-	<b>Training Loop:</b> The model undergoes training for a predetermined number of epochs. During training, the Siamese Network learns to distinguish between AD and NC images.
-	<b>Model Saving:</b> After training, the model is saved to a file (<b>SNN.pth</b>), allowing for future use.
-	<b>Loss Plotting:</b> A plot of training loss is generated to visualize the training progress.
</p>
<b><li>Making Predictions</li></b>
<p>The prediction phase is executed using the <b>predict.py</b> script. The main components of this phase include:

-	<b>Model Loading:</b> The trained Siamese Network model is loaded from the saved file.
-	<b>Device Selection:</b> Like the training phase, the code determines the device for execution (GPU or CPU).
-	<b>Predictions:</b> The model is used to make predictions on test data, determining whether an image is more likely to be AD or NC.
-	<b>Output Display:</b> Predictions are displayed for each sample, allowing an assessment of the model's performance.
</p>
</ol>

## 👽 Detailed Overall Working <a name = "detailed_overall_working"></a>
<h3>Model:</h3>
<p>A Siamese Neural Network is a form of a twin network, consisting of two identical subnetworks. The networks have the same configurations, parameters, and weights, hence identical. A pair of images is passed through each network, thus computing the features of each image. The similarity is then computed by passing the feature pairs through a euclidean distance function. The aim of the network is to produce a value such that we know if the images are different or the same, with 0 for different, 1 for same.</p>
<h3>Triple Loss:</h3>
<p>Due to the nature of SNN's, cross entropy functions do not work. Given this there are other types of loss functions used with SNN's. There are two types that are most used.</p>
<h3>Contrastive Loss:</h3>
<p>Contrastive loss simply takes two inputs and computes the euclidean distance between the image features. Points with similar features will have a low euclidean distance, and high euclidean distance for points with dissimilar features.</p>
<h3>Problem Statement:</h3>
<p>In this report we are attempting to classify the presence of Alzheimer's disease through the inspection of a brain scan.
It is possible to classify a brain scan as either Alzheimer's Disease (AD) or Cognitive Normal (CN) using SNN. To do this we need a dataset of image pairs to train the model. The pairs will be either two CN’s or a CN and an AD and have a corresponding label (0 for two CN’s, 1 for combination). After training the model, given a pair of images, the model can then either classify the pairs as either two CN’s or a combination of an CN and an AD.
At this point we can only tell if two images are CN or a combination of CN and AD. However, we want to have a model that we can pass a singular scan into and know whether the scan has AD or is a CN. To accomplish this, we take the trained subnetwork, and retrain the subnetwork with a classifier. Once trained this model can classify a brain scan.
</p>
<h3>Data Set:</h3>
<h3>Classification Data:</h3>
<h3>Model Architecture:</h3>
<h4>SNN:</h4>
<h4>Classification:</h4>
<h3>Results:</h3>

## 🙏🏽 Conclusion <a name = "conclusion"></a>
## 🔗 References <a name = "references"></a>