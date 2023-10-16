import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from dataset import OASISDataLoader
import modules
# import train

VQVAE_PATH = "./vqvae_model.txt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32

model = torch.load(VQVAE_PATH)
_, test_loader, _ = OASISDataLoader(batch_size=batch_size).get_dataloaders()

fig, ax = plt.subplots(1, 3, figsize=(8, 3))
for x in ax.ravel():
    x.set_axis_off()

test_real = next(iter(test_loader))  # load some from test data loader
test_real = test_real[0]
test_real = test_real.to(device).view(-1, 1, 128, 128).detach()

_, decoded_img = model(test_real)
decoded_img = decoded_img.view(-1, 1, 128, 128).to(device).detach()
real_grid = torchvision.utils.make_grid(test_real, normalize=True)
decoded_grid = torchvision.utils.make_grid(decoded_img, normalize=True)
decoded_grid = decoded_grid.to("cpu").permute(1, 2, 0)
real_grid = real_grid.to("cpu").permute(1, 2, 0)

pre_conv = (model.pre_vq_conv(model.encoder(test_real)))
_, test_quantized, _, indices = model.vq(pre_conv)
encoding = indices.view(32, 32)
encoding = encoding.to('cpu')
encoding = encoding.detach().numpy()


ax[0].imshow(real_grid)
ax[0].title.set_text("Real Image")
ax[1].imshow(encoding)
ax[1].title.set_text("Codebook Representation")
ax[2].imshow(decoded_grid)
ax[2].title.set_text("Decoded Image")
plt.savefig("Real vs decoded.png")
plt.show()