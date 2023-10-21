from imports import *

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Sourced from: https://huggingface.co/blog/annotated-diffusion
Author: Niels Rogge, Kashif Rasul
'''
# Define a schedule for beta values across timesteps
def beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

timesteps = 1000

# Define beta schedule
betas = beta_schedule(timesteps=timesteps)

# Define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# Calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# Calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# Extract time-dependent values from a tensor
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# Forward diffusion 
def q_sample(x_start, t, noise=None):
    if noise is None:
        # Generate noise
        noise = torch.randn_like(x_start)

    # Extract alpha values for the given timestep
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    # Perform the forward diffusion step
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

# Define the loss function
def get_loss(denoise_model, x_start, t, noise=None):
    if noise is None:
        # Generate noise
        noise = torch.randn_like(x_start)

    # Get the model's prediction of the noise
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    # Compute Smooth L1 loss between the true and predicted noise
    loss = F.smooth_l1_loss(noise, predicted_noise)

    return loss