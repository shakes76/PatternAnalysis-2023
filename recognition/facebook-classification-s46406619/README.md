# Semi-supervised node classification of the Facebook large page-page network dataset.

We use a multi-layer graph-convolutional network to classify the Facebook large page-page network dataset. [[1]](#1) Essentially, the given dataset consists of nodes (vertices) and edges between nodes. Each node represents an official Facebook page, and edges represent mutual likes between  two pages. Each node has 128 features associated with it, and belongs to one of four classes: politicians, government organizations, television shows, and companies. The task at hand is to be able to predict with high accuracy which of these four classes any given node in the dataset belongs to. 

Graph convolutional networks are an adaption of the typical neural network for graph-based data. As such, they orbit around the usage of the graph convolutional layer as introduced by Kipf et al. in 2016. [[2]](#2) This layer takes in its function the degree of each node $i$ and all the neighbours of $i$. To be exact, for a given node $i$, the graph convolutional layer computes the following:

$$ 
h_i=\sum_{j\in N_i}\frac{1}{\sqrt{\text{deg}(i)}\sqrt{\text{deg}(j)}}\boldsymbol{W}_{x_j},
$$ 

where $\boldsymbol{W}$ is a weight matrix, and $N_i$ is the set of neighbours (adjacent vertices) of $i$. Our network uses three graph convolutional layers followed by a final linear layer. Please see below a visualising of the overall network architecture.

![architecture](images/architecture.png?raw=true)

## Preprocessing & Training

The given dataset consists of 22470 128-dimensional feature vertices along with a list of 342004 bi-directional edges. No data preprocessing (i.e. dimensionality reduction) was performed upon the dataset whatsoever prior to training, so as to retain all the variance. A standard train/test split of 0.8/0.2 was used to gain accurate and worthwhile results.

The model completed training over 100 epochs in 44.46 seconds with a training accuracy of 96.79%. See below a graph of the model's training accuracy and loss over said 100 epochs.

![training](images/training.png?raw=true)

## Results

The model achieved an overall accuracy of 95.25% on the test set. Thus, overfitting was completely avoided. These results can be visualised through the use of t-SNE dimensionality reduction. Below we can see a 2D representation of the full dataset with labeled classes. Additionally, we see a 2D representation of the embeddings of the test set obtained by the model during prediction, with both the true and predicted labels coloured accordingly. 

![results](images/results.png?raw=true)

We can clearly see that the vertices of this dataset are highly clustered, and the model has learnt that effectively. Furthermore, the GCN model has emphasised the class clustering heavily by eliminating the vast majority of class outliers. Overall, we find that the the model has a robust performance on the test set, with a more than satisfactory test accuracy.

## Dependencies

* `numpy` 1.23.2
* `torch` 2.0.1
* `torch-geometric` 2.3.1
* `matplotlib` 3.5.3
* `sklearn` 1.1.2

Please note that `numpy.random.seed(42)` is set when performing the train/test split in `dataset.py`, and we set `random_state=1` when using the `TSNE` function from `sklearn.manifold`. Additionally, the `run.py` file is a simple test driver script that calls `train.py` and `predict.py` sequentially.

## References 

<a id="1">[1]</a> https://snap.stanford.edu/data/facebook-large-page-page-network.html

<a id="2">[2]</a> https://arxiv.org/abs/1609.02907
