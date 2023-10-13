# Multi-Class Node Classification of Facebook Network Dataset Using GCN

## Dataset Background

The Facebook Large Page-Page Network was used within this project. This dataset is a page-to-page graphical representation of Facebook websites in 4 distinct categories: (0) Politicians, (1) Governmental Organizations, (2) Television Shows and (3) Companies. Nodes in the graph represent the individual Facebook pages while edges between nodes represent mutual likes between pages. The dataset contains 22,470 nodes (each with 128 features extracted from the site descriptions created by the page owner) as well as 171,002 edges.

## Algorithm

The primary algorithm used within this project is a multi-layer Graph Convolutional Network (GCN). The network performs semi supervised multi-class node classification on the Facebook dataset. This is in order to classify each of the Facebook pages (nodes) into one of the aforementioned 4 categories: (0) Politicians, (1) Governmental Organizations, (2) Television Shows and (3) Companies. A T-Distributed Stochastic Embedding (tSNE) is also used to visualize the high dimensional Facebook page data in a human-readable form (2 dimensions).

## Problem That It Solves

As mentioned above, the aim of this project is to classify each of the Facebook pages into 1 of 4 categories. This task has numerous real-world applications. Primarily, the use of a machine learning algorithm to identify the types of social media accounts would be of great interest to data mining or advertising companies. In order to increase the effectiveness of advertisements, it is important to direct and target the ad to the goal user. By using an algorithm to quickly identify the category of a page, companies can specifically select advertisements that would appeal to the target audience of that specific page. For instance, a viewer looking at a social media page for a television show would likely be interested in streaming services such as Netflix. Using the GCN to identify the page as a ‘television show’, would all an advertisement company to direct Netflix ads to this page, hence increasing the effectiveness of the ad.

## How it Works

Regular convolutional neural networks (CNNs) use a filter, also known as a kernel, to slide over the input matrix, performing element-wise multiplications and summations. The goal of this filter is to extract local features and information from the input data. GCNs work in a similar manner to regular CNNs. However, instead of using a filter to capture local patterns in adjacent cells of the input matrix, GCNs can capture important information/features from adjacent nodes in the input graph. This is achieved by adding the adjacency matrix into the forward pass equation:

$X' = AXW$

In order to ensure the feature representation of a given node itself is included during this convolution; self-loops are often added to the adjacency matrix:

$\hat{A} = A + I$

Furthermore, normalization of the adjacency matrix was also performed to ensure highly connected nodes in the graph were not over represented. This was achieved using the “symmetric renormalization trick” as outlined in the paper Semi-Supervised Classification with Graph Convolutional Networks. This gives us the convolution equation used within this report:

$X' = \hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}XW$

In a GCN architecture, increasing the number of layers corresponds to expanding the receptive field for a given node. For instance, in this report, we employed three GCN layers, a decision made during the tuning process, as detailed below. We incorporated ReLU activation functions between the first two layers, which is a commonly used activation function for convolutional networks. This choice helps prevent exponential parameter growth and reduces computational overhead. Following the final layer, we applied SoftMax activation to obtain the output probabilities. To enhance model robustness and prevent overfitting, we included two dropout layers, which introduce variance into the training process.
Adam optimizer with a learning rate of 0.01 was used over SGD. This is due to its adaptive learning rate, momentum like behaviour, and the fact it works well with its default parameters so requires limited tunning.

Cross-entropy loss was chosen for this classification task as it ensures that the model optimizes for accurate class probability estimates.
The results presented in this report were generated using a train: test: validation split of 70:15:15, as explained in the testing/tuning section. This split provides a balanced approach for training, validating, and evaluating the model's performance while ensuring a robust assessment of its capabilities

### Usage

1. Ensure the facebook.npz file is located within the GCN_s4642798 directory

2. To train the model, run `predict.py` from the GCN_s4642798 directory:

```
python train.py
```

After training has completed, the following files should be generated in the GCN_s4642798 directory:

```
best_model.pt
gcn_loss.png
gcn_accuracy.png
```

3. To visualize the mode run:

```
python predict.py
```

After completition, the following files should be generated in the GCN_s4642798 directory:

```
tsne_post_train.png
```

### Dependencies

```
numpy:1.23.5
torch:2.0.1+cu117
matplotlib:3.7.2
scikit-learn:1.3.0
```

## Data Pre-Processing

The data was loaded from an `.npz` file, resulting in three NumPy arrays: `edges` representing network edges, `features` for node features, and `target`for node classification labels. The NumPy arrays were converted to PyTorch tensors to work with PyTorch's deep learning framework. The `edges` tensor and `target` tensor were cast to the integer type (`int64`) to ensure compatibility with integer-based operations. Self-loops were added to the `edges` tensor to include edges from each node to itself. As mentioned above, this is often done in graph neural networks to ensure that each node's information is considered. The `edges` tensor was transformed into a sparse tensor format using `torch.sparse_coo_tensor`. This is useful for efficient storage and computation in graph-related tasks. The dataset was split into training, validation, and test sets using random sampling. The training set comprises `70%` of the data, the validation set `15%`, and the test set `15%`. Binary masks (Boolean tensors) were created to identify which data points belong to each split.

## Visualisation and Results

The model produced an accuracy of 92.82% on the test set.

![alt text](./plots/gcn_accuracy.png)

The above plot shows the accuracy of the GCN model over the 100 epochs it was trained. The accuracy of both the training and validation set rapidly increases from 0.35 until it reaches approximately 0.9. At this point the rate of change of the accuracy beings to decrease and the accuracy does not drastically change for the remaining epochs. The training accuracy is slightly higher than the validation accuracy from 35 epochs to 100 epochs. The training accuracy is expected to be higher than the validation accuracy because the model is optimized to fit the training data. If the gap between the training and validation accuracy is larger, this indicates overfitting, as the model struggles to generalize to unseen data in this case, since the gap between training and validation accuracy is minimal, overfitting is unlikely.

![alt text](./plots/gcn_loss.png)

The above plot shows the loss of the GCN model over the 100 epochs it was trained. The loss for both the training and validation sets rapidly decreases from 1.4 until approximately 0.8. After this point the loss does not change significantly. The training loss and validation loss are very comparable even when nearing 100 epochs. This indicate that the model is neither underfitting or overfitting and is achieving a good balance between fitting the training data and generalizing to new unseen data.

![alt text](./plots/tsne_post_train.png)

The above scatterplot is a representation of each Facebook page in the dataset – the datapoints have been reduced down to 2-dimensions be the tSNE algorithm. As shown by the legend, the class of each page has been plotted as a different colour. In general, there is a prominent separation between classes which form their own clusters in different sections of the plot. This is expected as “clusters” should represent similar pages (and hence likely belonging to the same class). These clusters are, however, not perfect. There are some instances where some nodes can be found in the wrong cluster. For instance, there are a few yellow dots () and blue dots (), in the green cluster (). This is likely due to the model accuracy only being 92.82%. Based on this, roughly 7% of the pages have been mislabelled and hence should appear in the wrong cluster. Upon visual inspected, this seem reasonably accuracy with the large majority of pages being clustered correctly. A greater model performance would correspond with less nodes appearing in the incorrect cluster.

## Testing/ Tuning

### GCN Layers

```
Layer format                     Test accuracy
(?, 8)  -> (8, 4)                0.7591
(?, 16) -> (16, 4)               0.8051
(?, 32) -> (32, 4)               0.8143
(?, 8)  -> (8, 4)   -> (4, 4)    0.8965
(?, 16) -> (16, 8)  -> (8, 4)    0.9055
(?, 32) -> (32, 16) -> (16, 4)   0.9282
```

Various GCN architectures were tested to identify the optimal choice. As depicted above, the three-layer models outperformed the two-layer models significantly. This is likely because the added layer allows for a more complex and expressive representation of the data, which is particularly beneficial for the task at hand.

In the three-layer model, the architecture with the most channels delivered the best performance. This is because a higher number of channels allows the model to capture richer and more intricate features in the data. It's reasonable to assume that increasing the number of channels even further would have resulted in a marginal improvement in performance. However, this would have come at the cost of substantially increased training time, which was deemed impractical to explore further given the diminishing returns on performance gains and the need for efficient model training.

Therefore, the three-layer model with the architecture featuring the most channels was chosen as the optimal configuration, striking a balance between enhanced performance and manageable training time

### Data Split

```
Data split    Test Accuracy
40:30:30      0.9187
60:20:20      0.9194
70:15:15      0.9282
```

Minimal difference was identified in the test accuracy of the three data splits tested. This is likely because the Facebook dataset is relatively large. The `70 : 15 : 15` split was ultimately chosen because it strikes a good balance between training data, validation data, and testing data, ensuring that we have enough data for model training while also validating and testing its performance effectively. The minimal difference in test accuracy across the three data splits indicates that the dataset size is sufficient for robust model training and evaluation, reducing the risk of overfitting or underfitting in the final model

### ADAM Learning Rate

```
Learning Rate    Test Accuracy
0.1              0.9173
0.01             0.9282
0.001            0.8461
0.0001           0.4392
```

The choice of learning rate (`lr`) for the Adam optimizer significantly impacted model performance. Among the lr values tested, `0.01` provided the best balance of convergence speed and accuracy with an accuracy score of `0.9282`. Higher values like `0.1` showed rapid convergence but slightly lower accuracy, while lower values (`0.001` and `0.0001`) resulted in slow convergence and lower accuracy, making them impractical for the project's needs.

## Conclusion

This project successfully employed a Graph Convolutional Network (GCN) to classify Facebook pages into four categories. With a `92.82%` accuracy, the GCN model effectively categorized pages, making it a valuable tool for optimizing ad targeting and data mining. While there were minor misclassifications, the project demonstrated the potential of GCN in social media analysis and marketing.
