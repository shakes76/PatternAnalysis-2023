# Semi supervised node classification of the *Facebook Large Page-Page Network* dataset.

*Include a description of the algorithm and the problem that it solves
(approximately a paragraph), how it works in a paragraph and a figure/visualisation*

We use a multi-layer graph-covolutional network (GCN) to classify the Facebook Large Page-Page Network dataset. 

## Preprocessing & Training

The given dataset consists of 22470 128-dimensional feature vertices along with a list of 342004 bi-directional edges. No data preprocessing (i.e. dimensionality reduction) was performed upon the dataset whatsoever prior to training, so as to retain all the variance. A standard train/test split of 0.8/0.2 was used to gain accurate and worthwhile results.

The model completed training over 100 epochs in 44.46 seconds with a training accuracy of 96.79%. See below a graph of the model's training accuracy and loss over said 100 epochs.

![training](images/training.png?raw=true)

## Results

The model achieved an overall accuracy of 95.25% on the test set. Thus, overfitting was completely avoided. These results can be visualised through the use of t-SNE dimensionality reduction. Below we can see a 2D representation of the full dataset with labeled classes. Additionally, we see a 2D representation of the embeddings of the test set obtained by the model during prediction, with both the true and predicted labels coloured accordingly. 

![embeddings](images/embeddings.png?raw=true)

We can clearly see that the vertices of this dataset are highly clustered, and the model has learnt that effectively. Furthermore, the GCN model has emphasised the class clustering heavily, and thus eliminated the vast majority of class outliers.

## Dependencies

*It should also list any dependencies required, including versions and address reproduciblility of results, if applicable.*

* numpy 1.23.2
* torch 2.0.1
* torch-geometric 2.3.1
* matplotlib 3.5.3
* sklearn 1.1.2

Please note that `numpy.random.seed(42)` is set when performing the train/test split in `dataset.py`, and we set `random_state=1` when using the `TSNE` function from `sklearn.manifold`.