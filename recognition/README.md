# Semi Supervised node classification on Facebook Large Page-Page Network dataset
## Description
The facebook large page-page network dataset is a page-page graph of verified Facebook sites. Each node representa a Facebook page and the edges between nodes represent mutual likes between each page. The node features are descriptions of the page as written by the page owners, and have been embedded into an embedding space. The aim of this project is to classify each page into one of four categories: polititions facebook page, govermental facebook page, a tv show facebook page or a company page. 

The neural network is trained in a semi supervised manner. This models a real world problem where the data for the graph exists, but only a subset of nodes are labelled. In semi-supervised learning, the entire graph is used for training, but only the labeled nodes are used to compute the loss function. This way, the model learns from the graph structure and node features for the entire graph even though only a subset of node labels are available. 

## Graph neural networks and Graph Convolutional Networks
### Deep Learning Basics
In the simplest case, a deep neural network is a composition of parameterised linear functions $f^l$ for $l \in \{1, \dots, L\}$, each followed by an element-wise non-linear activation $\sigma$. 
```math
\begin{align}
    NN_\theta = f^L \circ \sigma \circ \dots \circ f^2 \circ \sigma \circ f^1.
\end{align}
```
Neural networks can be used as function approximates with learnable parameters $\theta$. The simplest neural network architecture is a multilayer perceptron, MLP where $f^l$ is defined to be 
```math
\begin{align}
    f^l(h) = W_l h + b_l
\end{align}
```
where $W_l$ is a matrix of learnable weights for the $l$th layer, $b_l$ is a learnable bias vector, and $h_l$ represents the feature vectors (or hidden representations) of the input data. For the activation function $\sigma$, a common choice is the rectified linear unit, or ReLU function, defined as 
```math
\begin{align}
    ReLU(h) = \max (0, h).
\end{align}
```
Neural networks are trained using the backpropagation algorithm and variants of the stochastic gradient descent algorithm to minimise a loss function $L$. The loss function $L$ is typically chosen to guide the network towards a specific output. In the case of multiclass classification with $C$ classes, the output layer of the neural network will have $C$ outputs, one for each class. The outputs are then passed through a softmax function which estimates the probability distribution over each class. The softmax is defined as
```math
\begin{align}
    \textrm{Softmax}(z_i) = \frac{exp(z_i)}{\sum_j exp(z_j)}
\end{align}
```
where $z_i$ represents the neural network output for class $i$. After estimating the probabilities using the softmax function, the loss function is typically chosen to be the cross-entropy loss, defined to be
```math
\begin{align}
    \textrm{CrossEntropyLoss}(y, \hat{y}) = -\sum_{c=1}^C y_c \log(p_c),\label{eqn: crossEntropyloss}
\end{align}
```
where $y_c$ is a binary indicator indicating the true class of the sample, and $p_c$ is the output softmax probability of the sample belonging to that class (\cite{hastie_neural_2009}).

### Graph Convolutional Networks
Graph neural networks are a neural network architecture built on top of graph structures. It takes in a graph $G = (V, E)$, where $V$ is a set of nodes $\{v_1, \dots, v_N\}$ and $E$ is a set of edges $(v_i, v_j)$ that connect the nodes $v_i$ and $v_j$. If for all $i, j$, $(v_i, v_j) \in E$ and $(v_j , v_i) \in E$, we say that the graph is undirected. Associated with each node is a $d$-dimensional feature vector $x_i$, where $d$ is the number of features for each node. Optionally each edge can have an associated feature vector $x_{(i, j)}$. 

The structure of the graph then determines the message-passing updates, which are executed in sequence to obtain updated node representations and edge representations. Let $h^l_i$ be the node representation of node $v_i$ on the $l^{th}$ iteration, and let $h^l_{(i, j)}$ be the edge representation after the $l^{th}$ update. Set $h^0_i = x_i$ and $h^0_{(i,j)} = x_{(i,j)}$.
```math
\begin{align}
    h_{(i, j)}^{l+1} &= f_{edge}(h_i^l, h_j^l, x_(i, j))\\
    h_i^{l+1} &= f_{node}(h_i, \sum_{h_i \in N(v_i)}h_{(j, i)}^l, x_i)
\end{align}
```
where $N(v_i)$ is the neighbourhood of $v_i$, that is all the nodes $v_j$ that $v_i$ is connected to. The update to the network is called a message-passing operation. Since each message-passing operation collects information from each of the neighbours, more message-passing operations aggregate information from further along the graph.

For a graph convolutional network, the message passing is done through the following function:
```math
\begin{align}
    h_i^{(l+1)} = \sigma\left(W_0^{(l)T} h_i^{(l)} + \sum_{v_i \in N(v_i)} c_{i, j} W_1^{(l)T} h_j^{(l)}\right)
\end{align}
```
where $\sigma$ is a non-linear activation function such as the element-wise ReLU function, $W_0^{(l)}$ and $W_1^{(l)}$ are learnable $d_l \times d_{d+1}$ parameter matrices and $c_{i, j} = 1/\sqrt{D_{i,i}D_{j,j}}$ is a normalisation constant, where $D_{i,i}$ is the degree of node $v_i$. The final node representations $h^L$ can then be used for regression or node classification problems in the normal way described above. 

## Training

## Validation

## T-SNE
