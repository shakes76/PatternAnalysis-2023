
# Classificaion of Alzheimer’s disease with Siamese network

This project shows how a Siamese network can be used to determine the similarity between two images of Alzheimer’s disease. 

## Siamese network
In the early 1990s, the Siamese network was presented, which can solve the signature verification problem. The model consists of two identical neural networks. This means that the weights and the number of different layers are the same for the left and right neural networks. At the end of this network, they are connected, and the L1 norm or L2 norm is calculated as distance. This layer is also called the distance layer. After the distance layer, the network uses a dense layer with a sigmoid function. The output of this dense layer is the result. The aim of this network is to determine the similarity between two input images.
An example image of the Siamese network is shown in the image below:

![AD!](\Images\AD.jpeg)
