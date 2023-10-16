"""
Author: William Barker
SN: 4642798
This script utilizes t-distributed Stochastic Neighbor Embedding (t-SNE) to reduce the 
dimensionality of the Facebook dataset and generate a visual plot.
"""
import dataset
import modules
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def plot_tsne(name, data, labels):
    """
    Funciton which uses tSNE to reduce dimensionality of facebook data
    and generate plot.
    """
    # peform tSNE dimensionality reduction
    z = TSNE(n_components=2).fit_transform(data.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    # scatter plot each class seperately to create legend
    for i in range(dataset.number_classes):
        indices = np.where(labels == i)
        indices = indices[0]
        plt.scatter(z[indices, 0], z[indices, 1], label=i)

    plt.title("tSNE Visualised")
    plt.legend()
    filename = "plots/{}.png".format(name)
    plt.savefig(filename)


def visulize():
    """
    Function to visualize the t-SNE plot using the trained GCN model and dataset.
    """
    # load model from saved file
    model = modules.GCN(
        dataset.sample_size, dataset.number_features, dataset.number_classes, 16
    )
    model.load_state_dict(torch.load("best_model.pt"))

    # set model evaluation mode and peform forward pass
    model.eval()
    out = model(dataset.X, dataset.edges_sparse)

    # plot and save t-SNE plot
    plot_tsne("tsne_post_train", out, dataset.y)


visulize()
