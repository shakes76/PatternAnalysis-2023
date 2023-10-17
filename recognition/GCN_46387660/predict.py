import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Placeholder for now
PATH = "~"

dataset = dataset.data


model = torch.load(PATH)
model.eval()

out = model(dataset.x, dataset.edge_index)
z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())
plt.figure(figsize(10,10))
plt.xticks([])
plt.yticks([])

plt.scatter(z[:,0], z[:, 1], s=70, c=dataset.y, cmap="Set2")
plt.show()