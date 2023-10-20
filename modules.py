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
    
class UNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(UNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = self._sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # Encoder
        self.enc1 = self._make_time_embedding(time_emb_dim, 1)
        self.block1 = nn.Sequential(
            Block((1, 224, 224), 1, 10),
            Block((10, 224, 224), 10, 10),
        )
        self.down1 = nn.Conv2d(10, 20, 4, 2, 1)

        self.enc2 = self._make_time_embedding(time_emb_dim, 20)
        self.block2 = nn.Sequential(
            Block((20, 112, 112), 20, 20),
        )
        self.down2 = nn.Conv2d(20, 40, 4, 2, 1)

        self.enc3 = self._make_time_embedding(time_emb_dim, 40)
        self.block3 = nn.Sequential(
            Block((40, 56, 56), 40, 40),
        )
        self.down3 = nn.Conv2d(40, 80, 4, 2, 1)

        self.enc4 = self._make_time_embedding(time_emb_dim, 80)
        self.block4 = nn.Sequential(
            Block((80, 28, 28), 80, 80),
        )
        self.down4 = nn.Conv2d(80, 160, 4, 2, 1)

        self.enc5 = self._make_time_embedding(time_emb_dim, 160)
        self.block5 = nn.Sequential(
            Block((160, 14, 14), 160, 160),
        )
        self.down5 = nn.Conv2d(160, 320, 4, 2, 1)

        # Bottleneck
        self.enc_mid = self._make_time_embedding(time_emb_dim, 320)
        self.block_mid = nn.Sequential(
            Block((320, 7, 7), 320, 320),
            Block((320, 7, 7), 320, 320),
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(320, 160, 4, 2, 1)
        self.dec1 = self._make_time_embedding(time_emb_dim, 320)
        self.block6 = nn.Sequential(
            Block((320, 14, 14), 320, 160),
        )

        self.up2 = nn.ConvTranspose2d(160, 80, 4, 2, 1)
        self.dec2 = self._make_time_embedding(time_emb_dim, 160)
        self.block7 = nn.Sequential(
            Block((160, 28, 28), 160, 80),
        )

        self.up3 = nn.ConvTranspose2d(80, 40, 4, 2, 1)
        self.dec3 = self._make_time_embedding(time_emb_dim, 80)
        self.block8 = nn.Sequential(
            Block((80, 56, 56), 80, 40),
        )

        self.up4 = nn.ConvTranspose2d(40, 20, 4, 2, 1)
        self.dec4 = self._make_time_embedding(time_emb_dim, 40)
        self.block9 = nn.Sequential(
            Block((40, 112, 112), 40, 20),
        )

        self.up5 = nn.ConvTranspose2d(20, 10, 4, 2, 1)
        self.dec_out = self._make_time_embedding(time_emb_dim, 20)
        self.block_out = nn.Sequential(
            Block((20, 224, 224), 20, 10),
        )

        self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)

    def forward(self, x, t):
        t = self.time_embed(t)
        n = len(x)

        out1 = self.block1(x + self.enc1(t).reshape(n, -1, 1, 1))  # (N, 10, 224, 224)
        out2 = self.block2(self.down1(out1) + self.enc2(t).reshape(n, -1, 1, 1))  # (N, 20, 112, 112)
        out3 = self.block3(self.down2(out2) + self.enc3(t).reshape(n, -1, 1, 1))  # (N, 40, 56, 56)
        out4 = self.block4(self.down3(out3) + self.enc4(t).reshape(n, -1, 1, 1))  # (N, 80, 28, 28)
        out5 = self.block5(self.down4(out4) + self.enc5(t).reshape(n, -1, 1, 1))  # (N, 160, 14, 14)

        out_mid = self.block_mid(self.down5(out5) + self.enc_mid(t).reshape(n, -1, 1, 1))  # (N, 320, 7, 7)

        out6 = self.block6(torch.cat((out5, self.up1(out_mid)), dim=1) + self.dec1(t).reshape(n, -1, 1, 1))  # (N, 160, 14, 14)
        out7 = self.block7(torch.cat((out4, self.up2(out6)), dim=1) + self.dec2(t).reshape(n, -1, 1, 1))  # (N, 80, 28, 28)
        out8 = self.block8(torch.cat((out3, self.up3(out7)), dim=1) + self.dec3(t).reshape(n, -1, 1, 1))  # (N, 40, 56, 56)
        out9 = self.block9(torch.cat((out2, self.up4(out8)), dim=1) + self.dec4(t).reshape(n, -1, 1, 1))  # (N, 20, 112, 112)

        out = self.block_out(torch.cat((out1, self.up5(out9)), dim=1) + self.dec_out(t).reshape(n, -1, 1, 1))  # (N, 10, 224, 224)
        out = self.conv_out(out)

        return out

    def _make_time_embedding_block(self, time_emb_dim, in_channels, out_channels):
        return nn.Sequential(
            self._make_time_embedding(time_emb_dim, in_channels),
            Block(in_channels, out_channels),
            Block(out_channels, out_channels)
        )

    def _make_time_embedding(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )
      
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
