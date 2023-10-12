# Multi-layer GCN Model (s4696681)

This folder contains the implementation of a semi-supervised multi-layer GCN (Graph Convolutional Network) for Facebook's Large Page-Page Network dataset 
Hyperlink to dataset: https://snap.stanford.edu/data/facebook-large-page-page-network.html


## Overview
Semi-supervised Graph Convolutional Networks (GCNs) for node classification leverage labeled and unlabeled data within a graph to classify nodes. They operate by learning a function that maps a node's features and its topological structure within the graph to an output label. During training, the model uses the available labels in a limited subset of nodes to optimize the classification function, while also considering the graph structure and feature similarity among neighboring nodes. This approach allows the model to generalize and predict labels for unseen or unlabeled nodes in the graph, enhancing performance particularly when labeled data are scarce. 

The Facebook Large Page-Page Network dataset has nodes that represent official Facebook sites, with the edges being mutual likes between the sites. The Nodes are classified into one of four categories, these being: politicians, governmental organizations, television shows, and companies. By leveraging both node feature information (from site descriptions) and topological information (from mutual likes between pages), the GCN can exploit local graph structures and shared features to infer the category of a page. This classification allows for a more efficient organization, retrieval, or understanding of the pages without manually labeling each one. In essence, it enables the automatic categorization of Facebook pages based on both their content and their relationships with other pages.



## Libray packages and dependencies


## Example inputs, outputs and plots of algorithm
[Provide steps or commands to run the model]


