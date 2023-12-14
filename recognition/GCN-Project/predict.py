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
    umap_embeddings = umap.UMAP(n_neighbors=30, min_dist=0.5, n_components=2).fit_transform(embeddings)
    true_labels = graph.ndata['Target'].numpy()

    # Create a UMAP embeddings plot with ground truth labels in colours
    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=true_labels, cmap='viridis', s=10)
    plt.colorbar()
    
    plt.savefig("Model.png")
    #plt.show()

'''
print_results(model, graph, test_mask, epoch, loss):
Prints out the epoch, loss and accuracy of the model.

accuracy: current accuracy of the model
epoch: epoch number the model is on
loss: current loss function of the model
'''
def print_results(accuracy, epoch, loss):
    print(f'Epoch {(epoch)}: Loss {loss.item()}, Test Accuracy {accuracy.item()}')