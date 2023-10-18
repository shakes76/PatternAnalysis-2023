import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import dataset
import modules

transform = TSNE

# Placeholder for now
PATH = "model.pt"

data = dataset.dataset
print("data retrieved")


model = torch.load(PATH)
print("model loaded")
model.eval()

out = model(data.x, data.edge_index)
print("computed out")
z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())
print("computed z")

plt.xticks([])
plt.yticks([])
# s changes the size of the markers
plt.scatter(z[:,0], z[:, 1], s=35, c=data.y, marker=".",cmap="Set2")
plt.show()