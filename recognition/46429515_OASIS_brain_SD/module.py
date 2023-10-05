import torch
import torch.nn.functional as F
from torch import nn
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Code referenced from: 
# https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL
# https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb

# Noise Scheduler (Forward Process)

def quadratic_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start**0.5, end**0.5, timesteps)**2

def get_index_from_list(vals, t, x_shape):
    """
    Helper function that returns specific index of t of a list of values
    vals while considering batch dimension
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device=device):
    """
    Takes image and timestep as input to return the image w/ noise
    q(x_t|x_0) = N(x_t;sqrt(alpha_t)*x_0, (1-alpha_t)I)
    x_0 is initial image, x_t is the noisy image at timestep t
    x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

# Define beta schedule
T = 500
betas = quadratic_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


# U-Net (Backwards Process)

# Aim is to obtain x_0, from x_t

class Block(nn.Module):
    def __init__(self, in_channel, out_channel, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channel)
        # UNet section (for downsampling/upsampling)
        if up:
            self.conv1 = nn.Conv2d(2 * in_channel, out_channel, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_channel, out_channel, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
            self.transform = nn.Conv2d(out_channel, out_channel, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_channel)
        self.bnorm2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        
    def forward(self, x, t, ):
        # First convolution
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimension
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
        # Add time channel
        h = h + time_emb
        # Second Conv - Provides representation of both time step + image information
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down/Upsample
        return self.transform(h)
    
    
class PositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # Check ordering here        
        return embeddings


class UNet(nn.Module):
    """
    Simple UNet model for OASIS brain dataset
    """
    def __init__(self):
        super().__init__()
        img_ch = 1
        down_ch = (32, 64, 128, 256, 512, 1024)
        up_ch = (1024, 512, 256, 128, 64, 32)
        out_dim = 1
        time_emb_dim = 32
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Initial projection
        self.conv0 = nn.Conv2d(img_ch, down_ch[0], 3, padding=1)
        
        # Downsampling
        self.downsample = nn.ModuleList([Block(down_ch[i], down_ch[i + 1], 
                                               time_emb_dim=time_emb_dim)
                                         for i in range(len(down_ch) - 1)])
        
        # Upsampling
        self.upsample = nn.ModuleList([Block(up_ch[i], up_ch[i + 1], 
                                               time_emb_dim=time_emb_dim, up=True)
                                         for i in range(len(up_ch) - 1)])
        
        self.output = nn.Conv2d(up_ch[-1], out_dim, 1)
        
    def forward(self, x, timestep):
        # Embed time
        t = self.time_mlp(timestep)
        # Initial convolution
        x = self.conv0(x)
        # U-Net input
        residual = []
        # downsample
        for down in self.downsample:
            x = down(x, t)
            residual.append(x)
        # upsample
        for up in self.upsample:
            # Add residual x to input
            residual_x = residual.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)
    
