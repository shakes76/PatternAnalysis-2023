# Multi-layer GCN on Facebook Large Page-Page Network dataset 
### student number: 46905774
### student name: Yangbo Liao

task: Create a suitable multi-layer GCN model to carry out a semi supervised multi-class node classification 
using Facebook Large Page-Page Network dataset with reasonable accuracy. (You may use this partially processed dataset 
where the features are in the form of 128 dim vectors.) You should also include a TSNE or UMAP embeddings plot with ground truth 
in colors and provide a brief interpretation/discussion.


## dataset information
This webgraph is a page-page graph of verified Facebook sites. Nodes represent official Facebook
pages while the links are mutual likes between sites. Node features are extracted from the site 
descriptions that the page owners created to summarize the purpose of the site.

This graph restricted to pages from 4 categories which are defined by Facebook. 
These categories are: politicians, governmental organizations, television shows and companies.
the Facebook Large Page-Page Network dataset which containing 22470 nodes, 
each with 128 unique features, and 171002 edges, representing connections in the network. The nodes fall into one of four distinct classes.

## algorithm
The algorithm employed in this project is a Graph Convolutional Network (GCN), a seminal architecture in 
the field of graph neural networks. The algorithm extends the concept of 
[convolutional neural networks (CNNs)](https://saturncloud.io/blog/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way/) to graph-structured data, accounting for the dependency of nodes through their 
connections. The multi-layer Graph Convolutional Network (GCN) model initiates with an Input Layer that takes a 
Node Feature Matrix X with dimensions [22470, 128] and an Adjacency Matrix Adj of dimensions [22470, 22470]. 
This is followed by four sequential GCN Convolution Layers. Each layer performs operations involving the 
multiplication of input features with a weight matrix, addition of a bias, and subsequent normalization. 
The ReLU activation function is applied to introduce non-linearity, and the processed features are then optionally 
subjected to dropout for regularization and batch normalization for stabilizing the learning process. 
The first layer outputs hidden features with dimensions [22470, 2*hidden_channels], while the subsequent layers output 
hidden features of [22470, hidden_channels]. The final convolution layer produces class scores with dimensions [22470, classes_size]. 
The model concludes with an Output Layer employing a LogSoftmax function, delivering a Class Probability Distribution with dimensions [22470, 4]. 
Throughout these layers, the transformation of features is facilitated by the adjacency matrix, ensuring the model captures the graph structure effectively. 
The process diagram is shown below

![image issue](https://github.com/Amberfafa/PatternAnalysis-2023/blob/topic-recognition/recognition/46905774_GCN/results_visualization/GCN_module_dia.png)

## problem that it solves
The aim of this project is to categorize each Facebook page into one of four categories.
This multi_layer GCN model is designed to address the challenge of semi-supervised node classification within the 
Facebook Large Page-Page Network dataset. The dataset is a representation of Facebook pages as nodes in a graph, 
with edges indicating their mutual connections. Each node possesses features, and they are classified into various categories. 
However, labels are available for only a small subset of nodes. 
GCN uses labeled instances and the inherent structure of the graph to predict the class of unlabeled nodes.


## How it works
The GCN operates by first initializing node representations corresponding to their features. 
It then iteratively updates these representations by aggregating feature information from neighboring nodes via convolutional layers. 
In this project, multiple such layers are used, each learning and extracting different levels of features from the node's neighborhood. 
Post feature aggregation, standard neural network components like ReLU , dropout , and batch normalization. 
The final layer of the GCN is a softmax classifier that assigns probabilities for each class to the nodes. 
During training, the model learns to classify nodes by minimizing the cross-entropy loss and smoothes the cross-entropy loss. 
Through back propagation and optimization techniques, model parameters can be adjusted.

## pre-processing
Preprocessing consisted of loading the data with PyTorch and adding self-loops to improve stability. The dataset has 22,470 samples and is divided into 70% training set, 
15% validation set and 15% test set, which is a standard division ratio often used for various deep learning dataset divisions.

## dependencies
- Python
- Pytorch
- Matplotlib
- Numpy
- Sklearn.manifold

## how to run the code
It is recommended that the 46905774_GCN folder be packaged into colab and run them, as utilizing the platform's GPU computing can greatly increase the speed of the computation
1. Make sure to download the facebook.npz file, and in the `dataset.py` file, change the dataset path to your own path
2. run the `train.py` to train the model
   ```
   %run train.py
   ```
   and we can get two plots of accuaracy and loss across epoches, which are shown in `results_visualization` folder
   named `gcn_accuracy.png` and `gcn_loss.png`
4. run the `predict.py` to plot embeddings with ground truth in colors
   ```
   %run predict.py
   ```
   These two plots are also shown in `results_visualization` folder
   named `pre_train.png` and `post_train.png`
   
## results and analysis
![image showing issue](https://github.com/Amberfafa/PatternAnalysis-2023/blob/topic-recognition/recognition/46905774_GCN/results_visualization/best_accuracy.png)!

After 800 epoches training of the model on training set we got 93.33% accuracy on test set

![image showing issue](https://github.com/Amberfafa/PatternAnalysis-2023/blob/topic-recognition/recognition/46905774_GCN/results_visualization/gcn_accuracy.png)

This image displays the accuracy of a model on training set and validation set over a series of epochs. As we can see the model 
quickly improve its accuracy during the initial epochs and then it tends to level off after a certain point. 
The validation and training accuracies are closely aligned, which means that the model isn't overfitting to the training set.

![image showing issue](https://github.com/Amberfafa/PatternAnalysis-2023/blob/topic-recognition/recognition/46905774_GCN/results_visualization/gcn_loss.png)

Since the difference in the loss values between the training set and the validation set is too large in the initial period,
if I show all 800 epochs of the image, we can't clearly see the difference between the two curves, so I only show the loss values of the first 15 epochs.
From the image we can know that The validation loss starts at a very high value but rapidly decreases and stabilizes. The training loss follows a similar trend, 
though it levels out at a slightly lower value than the validation loss. And the difference between these two lines proves that the model isn't overfitting.

![image showing issue](https://github.com/Amberfafa/PatternAnalysis-2023/blob/topic-recognition/recognition/46905774_GCN/results_visualization/pre_train.png)

I use t-SNE to reduce high-dimensional data to 2 dimensions so that I can visualize it on a 2-dimensional plane.
We can visualise these embeddings as points on a plot, colored by their true subject labels. 
This is the ground-truth plot before training. After training the plot is shown below

![image showing issue](https://github.com/Amberfafa/PatternAnalysis-2023/blob/topic-recognition/recognition/46905774_GCN/results_visualization/post_train.png)

We can see that the plot shows good clustering, where nodes of a single colour are mostly grouped together. 
it approves that the model has learned useful information about the nodes based on their class.

**Reference:**[SEMI-SUPERVISED CLASSIFICATION WITH
GRAPH CONVOLUTIONAL NETWORKS](https://arxiv.org/pdf/1609.02907.pdf)


