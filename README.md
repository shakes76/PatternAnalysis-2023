# A demonstration of semi-supervised multi-class node classification using a GCN model
A semi-supervised node classification is classifying unlabeled nodes in an undireted graph where there is a feature
vector for each node and a set of labeled nodes[^1]. 
## The dataset
We demonstrate the usage of semi-supervised node classification using the Facebook large page-page network dataset[^2].
A processed dataset[^3] was used instead to train the model. The dataset consists of 128-dimension feature vector, a
(n,2) edge list and the node's label.

### Example of data:
Feature dimension
![image](https://github.com/ChocomintIce1/COMP3710-Demo3/assets/69633077/fcd6458d-4aef-4647-a852-a30842b830bc)
Edge list
![image](https://github.com/ChocomintIce1/COMP3710-Demo3/assets/69633077/63eaa703-7ea4-456f-a2dd-e2418aa5ce63)
Label list
![image](https://github.com/ChocomintIce1/COMP3710-Demo3/assets/69633077/7ca05462-f2a2-4e6c-baa9-11400d619824)


![Visual example of node classification](https://github.com/ChocomintIce1/COMP3710-Demo3/assets/69633077/f6822d8c-fe7d-493a-87c2-014e36d07d76)

[^1]: https://ojs.aaai.org/index.php/AAAI/article/view/17211/17018
[^2]: https://snap.stanford.edu/data/facebook-large-page-page-network.html
[^3]: https://graphmining.ai/datasets/ptg/facebook.npz
