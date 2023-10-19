# shows example use of the trained model, prints out any results
# Thomas Bennion s4696627
import torch as pt
import numpy as np
import dgl
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt

'''
show_graph(graph, model):
Creates a UMAP embeddings plot with ground truth in colours.

graph: dgl graph representing the dataset
model: GCN model
'''
def show_graph(graph, model):
    with pt.no_grad():
        embeddings = model(graph, graph.ndata['Features'])

    # Convert PyTorch embeddings to NumPy array
    embeddings = embeddings.numpy()

    # Use UMAP to reduce the dimensionality
    umap_embeddings = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2).fit_transform(embeddings)
    true_labels = graph.ndata['Target'].numpy()

    # Create a UMAP embeddings plot with ground truth labels in colors
    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=true_labels, cmap='viridis', s=10)
    plt.colorbar()
    
    plt.savefig("Model.png")
    #plt.show()

'''
evaluate(model, graph, test_mask, epoch, loss):
Prints out the epoch, loss and accuracy of the model by 
evaluating it against the test set.

model: model being evaluated
graph: dgl graph representing the dataset
test_mask: numpy array stating which values are for testing
epoch: epoch number the model is on
loss: the loss function
'''
def evaluate(model, graph, test_mask, epoch, loss):
    # Evaluate the model on the test set
    model.eval()
    with pt.no_grad():
        logits = model(graph, graph.ndata['Features'])
        predictions = logits.argmax(1)
        accuracy = ((predictions[test_mask] == graph.ndata['Target'][test_mask]).float()).mean()

    print(f'Epoch {epoch}: Loss {loss.item()}, Test Accuracy {accuracy.item()}')