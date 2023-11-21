##################################   utils.py   ##################################
import torch
import numpy as np

# Splitting the images into patches and sequentialising them
def patch(img, patches):
    c, h, w = img.shape
    patch = torch.zeros(patches ** 2, h * w * c // patches ** 2)
    patch_size = h // patches

    for i in range(patches):
        for j in range(patches):
            patch[i * patches + j] = img[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size].flatten()
    return patch

# positional embeddings
def position(length, dim):
    res = torch.ones(length, dim)
    for i in range(length):
        for j in range(dim):
            res[i][j] = np.sin(i / (10000 ** (j / dim))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / dim)))
    return res