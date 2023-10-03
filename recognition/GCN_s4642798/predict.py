import dataset
import modules
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color)
    plt.show()


# load model
model = modules.GCN(
    dataset.sample_size, dataset.number_features, dataset.number_classes, 16
)
model.load_state_dict(torch.load("recognition/GCN_s4642798/best_model.pt"))

model.eval()
out = model(dataset.X, dataset.edges_sparse)
visualize(out, color=dataset.y)
