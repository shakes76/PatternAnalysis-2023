# Facebook Large Page To Page Network Dataset GCN 
Author: Thivan Wijesooriya, s4638766

## Graph Convolutional Network (GCN) Algorithm
Graph convolutional networks (GCN) are a type of Neural Network that works on graph structures. Graphs consist of nodes which represent entities and edges which represent some sort of connection between the two entities that are connected by the edge. GCNs take a nodes features as well as all of its neighbours features' and combine them before using a neural network to process this information. This process is repeated for each layer in the model. This model consists of two layers and an output layer. This model's outputs are the result of node classification of the data inputed. 

## The Problem
The dataset that is used is a Facebook Large Page-Page Network. Each node is an official facebook page and the edges are mutual likes between sites. Each Node has 12 features which are extracted from the site descriptions that the page owners created to summarize the purpose of the site. The nodes were retricted to 4 categories (defined by facebook) when collected in 2017. These four categories are; politicians, governmental organisations, television shows and companies. 

## How the algorithm works
This model created in modules.py has two hidden layers and one output layer. This was decided as, more than 2 layers will often result in a performance decrease as seen in Table 4 of [this paper](https://www.mlgworkshop.org/2018/papers/MLG2018_paper_49.pdf). The output layer takes the outputed 256 channels from the second layer and outputs one of the 4 classes for each channel inputted. This model is trained with the Adam optimizer and cross entropy loss for the criterion. These were chosen as they are the standard choices for neural networks which are used for multi-class classification problems. 

## Preprocessing
A large majority of the preprocessing work was already done as the dataset was given as a npz file with 3 tensors; features, target and edges. Using the method load from numpy the features were added to the x variable and the target was added to the y variable. The edges were extracted from the npz file and then changed to a list of index tuples through the use of `.t().contiguous()`. These three tensors were then combined into the pytorch geometric data class to form the dataset. Masks were then constructed for the test data and the train data. It was decided that a 70/30 split will be used for train and test data due to the fact that studies show that the best results are obtained if this split is used as mentioned by [this report](https://scholarworks.utep.edu/cs_techrep/1209/). Seeing as the nodes were not organised in any particular order it was decided that the first 16000 nodes will be used for training and the rest of the nodes will be used for testing. To accomplish this a mask was formed by making a boolean list of size 22470 (the total number of nodes) where true means that it would be used for the training set and false means that it would not be used for the training, similarly the testing mask was created. 

## Inputs and Outputs
The dataset's three tensors are shown below.

![dataset_inputs](https://github.com/ThivanW/PatternAnalysis-2023/assets/140519988/5fc43a0b-c519-4de3-af87-6e2e30d7293e)

Both the edges and feature tensors are used as inputs for training whilst target is only used for evaluating the model after it has been trained with the test set. After training and testing the dataset the following output is returned.

![data_output](https://github.com/ThivanW/PatternAnalysis-2023/assets/140519988/10b744af-45c8-4412-b351-b97dc3ddba93)

This output dictates the coordinates in which the node exists in the plot shown below. 


## Visualisation
Running train.py outputted the following.

![epoch and loss](https://github.com/ThivanW/PatternAnalysis-2023/assets/140519988/ea1ae860-dbb6-4349-a97c-0ad8c2a7a670)

This output was also shown in the graph below. Note that points were created for eeach epoch and not one for every 10 epochs.

![Loss Function plot](https://github.com/ThivanW/PatternAnalysis-2023/assets/140519988/036f59eb-2556-4a18-a0ce-50a47ec76523)

From this it can be seen that the loss function has begun to platue and as such further epochs do not need to be considered. Running predict.py with this model results in the following figure below.

![Untrained_Trained_Models](https://github.com/ThivanW/PatternAnalysis-2023/assets/140519988/4b6b4a02-8a58-4a5c-b020-e60828e2ca93)

This figure above shows the untrained model to the left and the trained model to the right. It is clear to see that there is clear groups that have been formed. The colour of the node dictates the actual class of the node and the location of the node and its surrounding nodes dictate the prediction created by the model.


## Dependencies
In order to be able to run all the code included there are three commands that need to be run on the terminal in a clean conda environment which are shown below
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
conda install matplotlib
```
The first line installs pytorch version 2.1.0, the second line installs pytorch geometric around the previously installed pytorch version and the third line installs the most recent version of matplotlib.

## References
The code used in [this tutorial](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html) was manipulated in order to understand what was needed to be done to be able to construct the dataset and the model. Similarly it was used in order to understand how to train the model and use the model. 
The code shown in [this tutorial](https://www.datacamp.com/tutorial/comprehensive-introduction-graph-neural-networks-gnns-tutorial) along with documentation on TSNE was used in order to understand how to use TSNE and what the necessary inputs were for the model. 
