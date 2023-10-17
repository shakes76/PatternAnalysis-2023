# Classification of Alzheimer's disease of the ADNI Brain data using a Siamese Network.

Name: Ethan Pinto

Student Number: 46422860

## Description of Algorithm: 
A Siamese network consists of two identical neural networks with the same weights and parameters, each of which takes in an input images. The outputs of the networks are feature vectors which are then fed into a contrastive loss function, which calculates the similarity between the two images. The main objective of an SNN is to differentiate between two images. To classify an image into one of two classes, a Multi Layer Perceptron was trained to take in the output embedding from the Siamese Neural Network and classify it into one of two classes: Alzheimer's Disease (AD) or Normal Cognitive Ability (NC).

## Problem it Solves:
This project aims to solve the problem of detecting Alzheimer's disease in brain scans. 

## Diagram of Siamese Neural Network Classifier
The classifier is made up of two separate stages, the first is the Siamese Neural Network, which takes in an image and outputs an embedding, and the second is the multi-layer perceptron, which takes in an embedding and classifies it.

![](siamese_diagram.png)

INSERT MLP DIAGRAM

## Data Preprocessing

```
def foo():
    if not bar:
        return True
```



## How it Works:
* Include inputs/outputs


## Figure/Visualisation:

* Plots

## Describe pre-processing, justify training, validation and testing splits of data.


## References
