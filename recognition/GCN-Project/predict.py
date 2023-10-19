# shows example use of the trained model, prints out any results
# Thomas Bennion s4696627
import torch as pt
import numpy as np
import dgl
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt

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
    
    plt.show()