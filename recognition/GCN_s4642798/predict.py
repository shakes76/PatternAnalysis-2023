import dataset
import modules
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def plot_tsne(name, data, labels):
    z = TSNE(n_components=2).fit_transform(data.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    for i in range(dataset.number_classes):
        indices = np.where(labels == i)
        indices = indices[0]
        plt.scatter(z[indices, 0], z[indices, 1], label=i)

    plt.title("tSNE Visualised")
    plt.legend()
    filename = "{}.png".format(name)
    plt.savefig(filename)


def visulize():
    # load model
    model = modules.GCN(
        dataset.sample_size, dataset.number_features, dataset.number_classes, 16
    )

    model.load_state_dict(torch.load("best_model.pt"))

    model.eval()
    out = model(dataset.X, dataset.edges_sparse)
    plot_tsne("tsne_post_train", out, dataset.y)


visulize()
