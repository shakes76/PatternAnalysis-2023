import torch
import torch.nn as nn

class DiffusionProcess(nn.Module):
    def __init__(self, betas, num_steps):
        super(DiffusionProcess, self).__init__()
        self.betas = betas  # List of beta values for each time step
        self.num_steps = num_steps

    def forward(self, x):
        for step in range(self.num_steps):
            beta = self.betas[step]
            noise = torch.randn_like(x) * torch.sqrt(beta)
            x = x + noise
        return x

class DiffusionNetwork(nn.Module):
    def __init__(self):
        super(DiffusionNetwork, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(1, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        # Decoder
        self.dec1 = nn.ConvTranspose2d(128, 64, 3, padding=1)
        self.dec2 = nn.ConvTranspose2d(64, 1, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.enc1(x)
        x = self.enc2(x)
        # Decoder
        x = self.dec1(x)
        x = self.dec2(x)
        return x



