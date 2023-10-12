# Multi-Class Node Classification of Facebook Network Dataset Using GCN

## Dataset Background:

The Facebook Large Page-Page Network was used within this project. This dataset is a page-to-page graphical representation of Facebook websites in 4 distinct categories: (0) Politicians, (1) Governmental Organizations, (2) Television Shows and (3) Companies. Nodes in the graph represent the individual Facebook pages while edges between nodes represent mutual likes between pages. The dataset contains 22,470 nodes (each with 128 features extracted from the site descriptions created by the page owner) as well as 171,002 edges.

## Algorithm

The primary algorithm used within this project is a multi-layer Graphical Convolutional Network (GCN). The network performs semi supervised multi-class node classification on the Facebook dataset. This is in order to classify each of the Facebook pages (nodes) into one of the aforementioned 4 categories: (0) Politicians, (1) Governmental Organizations, (2) Television Shows and (3) Companies. A T-Distributed Stochastic Embedding (tSNE) is also used to visualize the high dimensional Facebook page data in a human-readable form (2 dimensions).

## Problem That It Solves

As mentioned above, the aim of this project is to classify each of the Facebook pages into 1 of 4 categories. This task has numerous real-world applications. Primarily, the use of a machine learning algorithm to identify the types of social media accounts would be of great interest to data mining or advertising companies. In order to increase the effectiveness of advertisements, it is important to direct and target the ad to the goal user. By using an algorithm to quickly identify the category of a page, companies can specifically select advertisements that would appeal to the target audience of that specific page. For instance, a viewer looking at a social media page for a television show would likely be interested in streaming services such as Netflix. Using the GCN to identify the page as a ‘television show’, would all an advertisement company to direct Netflix ads to this page, hence increasing the effectiveness of the ad.

## How it Works

### Usage

### Dependencies

## Visualisation and Results

## Data Pre-Processing

## Testing/ Tuning

## Conclusion
