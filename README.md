# A demonstration of semi-supervised multi-class node classification using a GCN model
A semi-supervised node classification is classifying unlabelled nodes in an undirected graph where there is a feature vector for each node 
and a set of labelled nodes. A semi-supervised classification works by training the model using the small amount of labelled data, Then the 
trained model makes predictions based on the labelled data. Then the model's prediction on unlabelled data is treated as pseudo-labels. 
This results in a larger set of labelled data that combines the original labelled data and the newly generated labelled data. The model 
is retrained using the larger set of data to improve its performance. Then this process is repeated many times.[^1].

Visual example of node classification[^4]
![Visual example of node classification](https://github.com/ChocomintIce1/COMP3710-Demo3/assets/69633077/f6822d8c-fe7d-493a-87c2-014e36d07d76)

To make the predictions, a graph convolutional network (GCN) is used where similar to a convolutional neural network (CNN), 
GCN learns features by inspecting adjacent nodes. GNNs aggregate the feature vectors, then pass the result to the dense layer
and apply the activation function, similar to a CNN[^5].

Visual example of a GCN[^6]

![gcn_web](https://github.com/ChocomintIce1/COMP3710-Demo3/assets/69633077/106b4c73-55c0-415a-a5b8-7e2075f1f125)



## The dataset
We demonstrate the usage of semi-supervised node classification using the Facebook large page-page network dataset[^2].
A processed dataset[^3] was used instead to train the model. The dataset consists of 128-dimension feature vector, a
(n,2) edge list and the node's label. The dataset was split into training (0.7), validation (0.2) and testing (0.1).
Training set had the majority of the dataset to ensure the model had sufficient data to train to reduce potential bias.
Then validation had 0.2 to capture the overall performance of the model. Testing had the smallest size to capture the
model's performance on unseen data.

### Example of data:
Feature dimension <br />
![image <br />](https://github.com/ChocomintIce1/COMP3710-Demo3/assets/69633077/fcd6458d-4aef-4647-a852-a30842b830bc)

Edge list <br />
![image](https://github.com/ChocomintIce1/COMP3710-Demo3/assets/69633077/63eaa703-7ea4-456f-a2dd-e2418aa5ce63)

Label list <br />
![image](https://github.com/ChocomintIce1/COMP3710-Demo3/assets/69633077/7ca05462-f2a2-4e6c-baa9-11400d619824)

Visual representation of the graph using TSNE embedding
![TSNE](https://github.com/ChocomintIce1/COMP3710-Demo3/assets/69633077/ee4459a1-5c1d-4a7d-a0be-1d289624c6f2)

## Dependencies used
* Pytorch: 2.0.0+cpu
* torch_geometric: 2.3.1
* numpy: 1.23.3
* pandas: 1.5.3
* skearn: 1.2.2
* matplotlib: 3.7.1

## Files explained
* Dataset.py: Containing the data loader for transforming the Facebook dataset into torch_geometric Dataset.
* modules.py: Contains the GAN model
* train.py: Contains the source code for training, validating, testing and saving your model. 
* predict.py: Shows example usage of your trained model.
* GAN.pt: The pre-trained saved model

## Example
To use the pre-trained, just load in the data and enter your data. 
```
import torch
import numpy as np
from torch_geometric.data import Data

...

data = Data(x=x, edge_index=edge_index, edge_attr=None, y=y) # Create geometric_torch dataset
model = torch.load('GCN.pt') # Load in the data
model.eval() # use evaluation mode
out = model(data.x, data.edge_index) # Predict model
```

## Training
### Training loss
128-100-64-32 layers <br />
![0 9453716065865598](https://github.com/ChocomintIce1/COMP3710-Demo3/assets/69633077/e932e1c6-f050-4227-a867-d9a05c6c3b5f)


128-64-16 layers <br />
![0 9461504227859368](https://github.com/ChocomintIce1/COMP3710-Demo3/assets/69633077/16239887-fd33-427b-8c3c-8f61dc2c9854)


128-64 layers <br />
![0 9362483311081442](https://github.com/ChocomintIce1/COMP3710-Demo3/assets/69633077/f294aa8a-73c0-4b17-b2cc-3ff2018cecc0)


128-16 layers <br />
![0 9224521584334668](https://github.com/ChocomintIce1/COMP3710-Demo3/assets/69633077/aa501314-6793-4046-887e-76953c88f7f2)

### Testing accuracy
| Layers       |   Testing accuracy    |
| :-----------: | :------------------: |
| 128-100-64-32 | 0.9453716065865598   |
| 128-64-16     | 0.9461504227859368   |
| 128-64        | 0.9362483311081442   |
| 128-16        | 0.9224521584334668   |

The table above shows how the model was trained and how the layers was chosen. The highest accuracy occurs when 128-64-16 layers are used. 
Adding an extra layer leads to slightly lower accuracy, possibly due to overfitting and the model is unable to adapt to new unseen data. 
Using less layers also results in lower accuracy. Further reducing number of hidden units leads to a lower accuracy. This is due to underfitting 
where the model is unable to capture the relationship between the features and labels accurately.

[^1]: https://ojs.aaai.org/index.php/AAAI/article/view/17211/17018
[^2]: https://snap.stanford.edu/data/facebook-large-page-page-network.html
[^3]: https://graphmining.ai/datasets/ptg/facebook.npz
[^4]: https://www.researchgate.net/figure/Illustration-of-semi-supervised-node-classification-Blue-and-green-denote-those-nodes_fig1_355873169
[^5]: https://www.datacamp.com/tutorial/comprehensive-introduction-graph-neural-networks-gnns-tutorial
[^6]: https://tkipf.github.io/graph-convolutional-networks/
