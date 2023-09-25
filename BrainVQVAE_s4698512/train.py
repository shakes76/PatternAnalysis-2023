"""
Hugo Burton
s4698512
20/09/2023

train.py
Contains source code for training, validating, testing and saving model.
Model imported from modules.py and data loader imported from dataset.py
Plots losses and metrics during training.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid

from modules import VectorQuantizedVAE


def train(data_loader, model, optimiser, device, beta, steps):
    # Loop over the images in the data loader
    for images, _ in data_loader:
        # Move the images in the batch to the GPU
        images = images.to(device)

        optimiser.zero_grad()

        x_til, z_e_x, z_q_x = model(images)

        # Reconstruction Loss
        recon_loss = F.mse_loss(x_til, images)

        # Vector Quantised Objective Function
        # We need to detach the gradient becuse gradients won't work in disctete space for backpropagation
        vq_loss = F.mse_loss(z_q_x, z_e_x.detach())

        # Commitment objective
        commit_loss = F.mse_loss(z_e_x, z_q_x.detach())

        fin_loss = recon_loss + beta * commit_loss
        fin_loss.backward()

        optimiser.step()
        steps += 1
