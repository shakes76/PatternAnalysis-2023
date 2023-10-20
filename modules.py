import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        return x
    
      
class DDPM_UNet(nn.Module):
    """
    Diffusion Denoising Probabilistic Model (DDPM) with a U-Net architecture.
    """
    def __init__(self, network, n_steps=200, min_beta=1e-4, max_beta=0.02, image_shape=(1, 224, 224), device=None,):
        super(DDPM_UNet, self).__init__()
        
        self.n_steps = n_steps
        self.device = device
        self.image_shape = image_shape
        self.network = network.to(device)
        
        # Compute the betas and alphas based on the given min and max betas
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        self.alphas = 1 - self.betas
        
        # Compute cumulative product of alphas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, original_image, time_step, noise=None):
        """
        Introduce noise to the original image based on the provided timestep `time_step`.
        """
        n, c, h, w = original_image.shape
        a_bar = self.alpha_bars[time_step]
        
        # If noise is not provided, generate one with the same shape as the original_image
        if noise is None:
            noise = torch.randn(n, c, h, w).to(self.device)
        
        # Compute the noisy version of the original_image
        noisy_image = (a_bar.sqrt().reshape(n, 1, 1, 1) * original_image) + ((1 - a_bar).sqrt().reshape(n, 1, 1, 1) * noise)
        return noisy_image

    def denoise(self, noisy_image, time_step):
        """
        Estimate the denoised version of the noisy image for the given timestep `time_step`.
        This runs the image through the U-Net architecture to return its estimation of the noise.
        """
        return self.network(noisy_image, time_step)
