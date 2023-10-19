# Classification of Alzheimer's disease of the ADNI Brain data using a Siamese Neural Network

Name: Ethan Pinto

Student Number: 46422860

## Problem Description:
The ADNI (Alzheimer's Disease Neuroimaging Initiative) Brain Dataset includes both MRI (Magnetic Resonance Imaging) and PET (Positron Emission Tomography) scans. By analysing these scans using deep learning models, we can effectively detect whether a patient has Alzheimer's disease. In this project, a Siamese Neural Network will be utilised to assist in the classification of brain scans into one of two categories: Alzheimer's Disease (AD) or Normal Cognitive Ability (NC).

## Description of Algorithm: 
A Siamese network consists of two identical neural networks with the same weights and parameters, each of which takes in an input images. The main objective of an SNN is to differentiate between two images i.e. provide a measure of similarity between the inputs. The outputs of the networks are feature vectors which are fed into a contrastive loss function, which calculates the similarity between the two images. To classify an image into one of two classes, a Multi Layer Perceptron was trained to take in the output embedding from the Siamese Neural Network and classify it into one of two classes: Alzheimer's Disease (AD) or Normal Cognitive Ability (NC).

## Diagram of Siamese Neural Network Classifier
The classifier is made up of two separate stages, the first is the Siamese Neural Network, which takes in a 3 x 128 x 128 image and outputs a 1D feature vector embedding of size 128. The second stages is the multi-layer perceptron, which takes in an embedding and classifies it.

![](siamese_diagram.png)

INSERT MLP DIAGRAM

## Dataset Structure
The ADNI dataset is currently stored on the Rangpur HPC and has the following file structure.

/home/groups/comp3710/<br>
&nbsp;&nbsp;&nbsp;&nbsp;->ADNI<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;->test<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;->AD<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;->NC<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;->train<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;->AD<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;->NC<br>

> There are a total of 9001 images in the test dataset, and 21521 images in the train dataset.

## Data Preprocessing

There are

```
def foo():
    if not bar:
        return True
```

## Training
The 


## How it Works:
* Include inputs/outputs


## Figure/Visualisation:

* Plots



## How to run code i.e. description of file.
run train.py with ...

## Dependencies (Main ones)


## References
