# COMP3710 Project: Siamese classification for ADNI data
**Student Number:** 46419712

## Table of Content
1. [Introduction](#1-introduction)
2. [Project structure](#2-project-structure)
3. [Reproducibility](#3-reproducibility)
4. [Model](#4-model)
5. [Train and validate loss](#5-train-and-validate-loss)
6. [Result](#6-result)
7. [Future improvement](#7-future-improvement)
8. [References](#8-references)

## 1. Introduction
The Siamese model is a powerful deep learning model that often used to assess dissimilarity between two images. In this project, this model will be adapted classify the ADNI dataset, determine whether a the brain image belongs to a patient with Alzheirmer's disease or a normal person.

## 2. Project structure
1. ```modules.py``` containing the code for the Siamese model, Siamese Contrastive Loss and Binary Classifier model.
2. ```dataset.py``` containing the data loader for loading and preprocessing ADNI data. This include split the train data to 80% training and 20% validating. This also inculude custom dataloader to handle PairDataset for Siamese model.
3. ```train.py``` containing the source code for training, validating, testing and saving the training model. The test result will be print after finish training.
4. ```predict.py``` plot the image and classify whether the image is belong to AD or NC class. This will be compare side by side with the actual label of the image.

## 3. Reproducibility
This project is reproducible, given that it using deterministic algorthm for the convolutional layer and the seed whenever random variable is used.

### 3.1. Dependencies
The dependencies of this project is install using miniconda. Here is a [link](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) on how to install miniconda.

The following is the dependencies and version of the dependencies.

| Dependency   | Version     |
| ------------ | ----------- |
| python       | 3.8.17      |
| pytorch      | 2.0.1       |
| torchvision  | 0.15.2      |
| matplotlib   | 3.7.2       |
| numpy        | 1.24.3      |
| scikit-learn | 1.0.2       |

### 3.2. Command to reproduce the result
Assuming all the relevant dependencies are installed.
Also, note that to reproduce the run, you will need to change the path in utils.py to match the path of your environment.

```python3 train.py```

When run the above command, it will first train the Siamese model. After that it will use the Siamese model to train the classifier model. And finally print out the accuracy of predicting the test data result in stdout. In addition, after finish running, in the "result" folder (you will need to create this folder or change the path in utils), it will return siamese.py and classifier.pt that contains the siamese and classifier model respectively. 


```python3 predict.py```

If the program is successully run, ```Siamese.pt and Classifier.pt``` exists, running predict.py will return the accuracy result in the stdout and save a random batch of test data that the model predicted the class in comparison to real class.

## 4. Model
### 4.1. Background
According to a paper by Koch, Zemel, and Salakhutdinov [1] about Siamese network for one shot image recognition, the Siamese model is best work when used to find the dissimilarity between two images. That is why, traditional Siamese model often train in a pair dataset.

* Siamese architecture

![Siamese oneshot](research_image/Siamese_oneshot.png)
The figure above demonstrate Siamese's architecture One-shot image recognition.

Based on the paper, key features from the Siamese's model One-shot image recognition includin, feature extraction using convolutional layer to extract feature vector from the image. This process is done with two identical subnetworks and compare between two iamges. The goal is to identify the dissimilarity between the two images.

After the first four layer of feature extraction, the distance metric (Siamese L1 distance) is calculated between the two images, and this will return the distance represent the dissimilarity. The image with similar dissimilarity will be closer to each other while higher dissimilarity will be push away. This is done using the contrastive loss function
$L = (1-Y) * ||x_i-y_j||^2 + (Y) * \max(0, m-||x_i-y_j||^2)$ .

### 4.2. Implementation

Adapting from the Siamese architecture above, the project is followed Siamese model architecture from the oneshot learning paper to implement the layer for Siamese model and the contrastive loss function for criterion.
![Siamese oneshot](research_image/Siamese_oneshot.png)

**Data preprocessing**

For data processing, there is no data for validating so, the train dataset was split in 80% for training and 20% for validating. To avoid data leakage, the data split was performed on patient level, so dataset that belong to the same patient will not shared between training and validating data.
In addition, the size of the image is resize to 105 to match with the classification from the paper. The result showed that when resize to 105, it not only train faster but the accuracy remain similar with little differences.

**Binary classification model**

Once the Siamese model is finish training, the model was used to help trained the binary classifier to classify the class of the image. First, the image will go through the Siamese to extract the feature vector. This feature vector is expected to distancing images that are dissimilar. The classifier is then used to classify the image. The classifier follow a simple multi fuly connected layer (5 fully connected layers) for better classification.


## 5. Train and validate loss

![Siamese loss plot](result/siamese_loss_plot.png)

The Siamese's train and validate loss show that when training and validating, the loss is converging and getting close to one another.

![Classifier loss plot](result/classifier_loss_plot.png)

However, when the image is put through the train Siamese model to extract feature vector to train the binary classifier, the loss model from the binary classifier showed that it is overfitting. This is due to the validate loss 

<div>
    <img src="result/tsn_train.png" alt="t-SNE train data" width="40%">
    <img src="result/tsn_validate.png" alt="t-SNE validate data" width="40%">
</div>

The t-SNE diagram shows that when evaluate Siamese model during validate using test set, there is some clear difference in the separation between the AD and NC data. However, in the train dataset used for training classifier, there is no clear dissimilarity between the AD and NC classes. This can be a good indication that there might be some overfitting in the training where for certain dataset, the Siamese model extract a good quality feature vector for differentiate between the two class, whereas for other cases, it doesn't work very well.

## 6. Result

The accuracy of the classifier model during training and validating further prove that there is overfitting in the model, where the accuracy of the train is growing while the accuracy of validate data capped at around 75%.
![Result Accuracy loss plot](result/Accuracy_plot.png)

### 6.1. What went wrong
Initially, this project has taken a wrong path where the file path for training dataset was accidentally use during the testing phase. This has cause false information and with the remaining time (2 days before the deadline), it is impossible to tune the hyper-parameters and the model to increase the accuracy.
When the wrong path was used, the highest accuracy achieved is 82.65%.
![img1](result/img1.png)

The image above show the visual prediction of the model, classifying the image from the test dataset vs the actual label.

### 6.2. Accuracy after correct the issue

After correcting the path for the testing phase, the actual accuracy is fell down to 62.4%.

Some attempted to tuning the accuracy include, using batchnorm and dropout layer to prevent overfitting from the Siamese model. However, in the end, the highest accuracy achieved is only 63.4%. Another attempt was using ResNet-18 as Siamese layer by remove the last layer and replace with the fully connected layer to extract the feature vector. However, the ResNet-18 doesn't work very well without careful tuning and the resource is limited in the remaining time, hence, the best accuracy achieved using ResNet-18 is only around 50.4%. 

## 7. Future improvement
Due to unfortunate incident that causing misleading in interpreting the model performance, future work will focus on tuning the parameters and apply more data augmentation to increase the accuracy of the model. In addition, this project will explore more on using ResNet-18 (CNN) as Siamese embedding layer to extract better feature vector from the data.

## 8. References
[1]	G. Koch, R. Zemel, and R. Salakhutdinov, "Siamese neural networks for one-shot image recognition," in ICML deep learning workshop, 2015, vol. 2, no. 1: Lille. 

**Code adaptation from external source is reference within the code comment**
