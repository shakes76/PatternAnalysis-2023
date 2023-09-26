from utils import *
from dataset import *

class VAE(nn.Module):
    """
    Variational Auto Encoder for OASIS_MRI
    """
    def __init__(self, input_dim, z_dim, h_dim):
        super().__init__()
        # encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)

        # output diagonal values of covariance matrix
        # assuming the pixels are conditionally independent 
        self.hid_2mu = nn.Linear(h_dim, z_dim)      # for mu
        self.hid_2sigma = nn.Linear(h_dim, z_dim)   # for stds (to compute log-variance)

        # decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.img_2hid(x)) # mu
        mu = self.hid_2mu(h)
        sigma = self.hid_2sigma(h)   # stds
        return mu, sigma

    def decode(self, z):
        new_h = F.relu(self.z_2hid(z))  # z: encoded
        x = torch.sigmoid(self.hid_2img(new_h))
        return x

    def forward(self, x):
        mu, sigma = self.encode(x)

        # Sample from latent distribution from encoder
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma * epsilon

        x = self.decode(z_reparametrized)
        return x, mu, sigma
