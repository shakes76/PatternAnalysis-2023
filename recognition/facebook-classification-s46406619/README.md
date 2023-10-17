# Semi supervised node classification of the *Facebook Large Page-Page Network* dataset.

*Include a description of the algorithm and the problem that it solves
(approximately a paragraph), how it works in a paragraph and a figure/visualisation*

We use a multi-layer graph-covolutional network (GCN) to classify the Facebook Large Page-Page Network dataset. 

## Results

See below a graph of the model's steady training accuracy and loss over 100 epochs.

![training](images/training.png?raw=true)

The model completed training after 100 epochs with a 96.79% accuracy on the training set. Moreover, the model achieved an accuracy of 95.25% on the test set. Thus, overfitting was avoided. These results were further visualised through the use of a t-SNE embedding. Both the original 128-dimensional dataset and the 16-dimensional embeddings produced after training the model were reduced via t-SNE dimensionality reduction to two dimensions. The results are shown below.

![labels](images/labels.png?raw=true)

We clearly see that the GCN model has efficiently clustered the dataset by class. 

## Preprocessing

*Describe any specific pre-processing you have used with references if any. Justify your training, validation and testing splits of the data.*

## Dependencies

*It should also list any dependencies required, including versions and address reproduciblility of results, if applicable.*