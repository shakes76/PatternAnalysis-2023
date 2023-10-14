import torch
import torch.nn.functional as F

#Modules utils

# Define beta schedule

def quadratic_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start**0.5, end**0.5, timesteps)**2

def get_index_from_list(values, t, x_shape):
    """
    Helper function that returns specific index of t of a list of values
    vals while considering batch dimension
    """
    batch_size = t.shape[0]
    out = values.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

T = 300
betas = quadratic_beta_schedule(timesteps=T)

# Pre-calculate terms for closed form equation
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)