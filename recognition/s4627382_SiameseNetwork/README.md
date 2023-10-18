# KNN classifier based on siamese network embedding
## Introduction
The purpose of this project is to construct a [Siamese network](#siamese-network) and use its embedding to train a [knn classifier](#k-nearest-neighbour) to classify the [Alzheimer's Disease Neuroimaging Initiative (ADNI)](#adni-dataset) brain dataset.

### Siamese Network
A Siamese network is a distance-based neural network. It consists of two weight-shared subnetworks and a designated loss function. The network takes two images as inputs, and then pass through their corresponding subnetworks for feature extraction. These subnetworks produce two flattened layers, called embeddings, which are then fed into the loss function. 
![Siamese Network Architecture](PatternAnalysis-2023/recognition/s4627382_SiameseNetwork/Images/SiameseNet.png)
In here, contrastive loss will be used. This loss function will calculate 

### K Nearest Neighbour

### ADNI Dataset

## Reference
https://medium.com/swlh/one-shot-learning-with-siamese-network-1c7404c35fda
