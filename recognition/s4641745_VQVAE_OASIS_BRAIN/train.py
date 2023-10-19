import os

import torch as t
import torch.nn as nn
import torch.utils.data
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from dataset import vqvae_test_loader, vqvae_train_loader, MODEL_PATH
from models import VQVAE
from utils import save_image

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
if not t.cuda.is_available():
    print("Warning CUDA not found. Using CPU")

# VQVAE Hyper params
LR_VQVAE = 1e-3
BATCH_SIZE_VQVAE = 32
MAX_EPOCHS_VQVAE = 4
NUM_HIDDENS = 128
RESIDUAL_INTER = 32
NUM_EMBEDDINGS = 512
EMBEDDING_DIM = 64
BETA = 0.25
DATA_VARIANCE = 0.0338

# create VQVAE model
model = VQVAE(NUM_HIDDENS, RESIDUAL_INTER, NUM_EMBEDDINGS, EMBEDDING_DIM, BETA)
model.to(device)

# Init optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR_VQVAE)
train_recon_loss = []

# train VQVAE
for i in range(MAX_EPOCHS_VQVAE):
    print(f"EPOCH [{i+1}/{MAX_EPOCHS_VQVAE}]")

    size = len(vqvae_train_loader.dataset)
    batch_losses = []
    i = 0
    for batch, (X, _) in enumerate(vqvae_train_loader):
        X = X.to(device)

        optimizer.zero_grad()
        vq_loss, data_recon = model(X)

        recon_error = F.mse_loss(data_recon, X) / DATA_VARIANCE
        loss = recon_error + vq_loss
        loss.backward()
        optimizer.step()
        batch_losses.append(recon_error.item())

        if (i+1) % 100 == 0:
            print(f"Step {i} -  recon_error: {np.mean(batch_losses[-100:])}")
        i += 1

    loss = sum(batch_losses) / len(batch_losses)

    train_recon_loss.append(loss)
    print(f"Reconstruction loss: {loss}")

# Save model
t.save(model, os.path.join(MODEL_PATH, "vqvae.txt"))

# save samples of real and test data
real_imgs = next(iter(vqvae_test_loader)) # load some from test dl
real = real_imgs[0]
real = real.to(device)
_, test_recon = model.forward(real) # forward pass through vqvae to create reconstruction
save_image(real, 'real-sample.png')
save_image(test_recon, 'recon-sample.png')