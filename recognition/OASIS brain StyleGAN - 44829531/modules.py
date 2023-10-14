import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 0.00001


class Noise(nn.Module):
    """
    Adds noise to the input. This helps in introducing 'randomness' to the generated images.

    b: A trainable parameter that scales the input noise.
    """
    def __init__(self):
        super(Noise, self).__init__()
        # Initialise b to be a random number.
        self.b = nn.Parameter(torch.randn(1, 1, 1, 1))

    def forward(self, x, noise):
        # Adds noise to the input.
        return x + self.b * noise


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalisation (AdaIN) is a technique that allows the model to learn the style of the image.

    param: epsilon: A small value to prevent division by zero.
    """
    def __init__(self, epsilon=EPSILON):
        super(AdaIN, self).__init__()
        self.epsilon = epsilon
        self.dense_y_scale = nn.Linear(256, 1)
        self.dense_y_bias = nn.Linear(256, 1)

    def forward(self, x, w):
        y_scale = self.dense_y_scale(w).view(-1, 1, 1, 1)
        y_bias = self.dense_y_bias(w).view(-1, 1, 1, 1)
        mean = x.mean(dim=[1, 2, 3], keepdim=True)
        stdev = x.std(dim=[1, 2, 3], keepdim=True)
        return y_scale * ((x - mean) / (stdev + self.epsilon)) + y_bias


class WNetwork(nn.Module):
    """
    The W network is a 8-layer fully connected network that takes in the latent vector z and outputs the style vector w.
    """
    def __init__(self, latent_dim=256):
        super(WNetwork, self).__init__()
        self.layers = nn.Sequential(*[nn.Sequential(nn.Linear(latent_dim, 256), nn.Linear(256, latent_dim), nn.LeakyReLU(0.2)) for _ in range(8)])

    def forward(self, z):
        return self.layers(z)
