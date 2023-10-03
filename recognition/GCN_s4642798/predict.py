import dataset
import modules
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize(name, h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color)
    filename = "{}.png".format(name)
    plt.savefig(filename)
    plt.show()


# load model
model = modules.GCN(
    dataset.sample_size, dataset.number_features, dataset.number_classes, 16
)

model.eval()
out = model(dataset.X, dataset.edges_sparse)
visualize("tsne_pre_train", out, color=dataset.y)

model.load_state_dict(torch.load("best_model.pt"))

model.eval()
out = model(dataset.X, dataset.edges_sparse)
visualize("tsne_post_train", out, color=dataset.y)
