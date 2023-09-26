import dataset
import module
import torch
import torchvision
import torch.nn.functional as F
from torch import nn
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import model from module.py
model = module.UNet()

## Code referenced from:
# https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL
# https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb

# Loss function
def get_loss(model, x_0, t):
    """
    Loss function using L1 loss (Mean Absolute Error)
    L_t (for random time step t given noise ~ N(0, I)):
    L_simple = E_(t,x_0,e)[||e - e_theta(x_t, t)||^2]
    where e is added noise, e_theta is predicted noise
    x_0: image
    """
    x_noise, noise = module.forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noise, t)
    return F.l1_loss(noise, noise_pred)

# Sampling
