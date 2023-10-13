import os
import module
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

## dataset.py

# Path to images
root_path = 'data/keras_png_slices_data'

# Define image size
IMAGE_SIZE = 128

# Define batch size of data
BATCH_SIZE = 32


## module.py

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

T = 500
betas = quadratic_beta_schedule(timesteps=T)

# Pre-calculate terms for closed form equation (in module.py)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


## train.py

# Output directory to save images
output_dir = './image_output'

# Number of Epochs for training
epochs = 800

# Loss function
def get_loss(model, x_0, t, device):
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
@torch.no_grad()
def sample_timestep(model, x, t):
    """
    Noise in the image x is predicted and returns denoised image
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        return model_mean # image w/o noise
    else:
        # add noise
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise
    

def save_tensor_image(image, epoch, step, output_dir):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage()
    ])
     
    # Shape image being saved so its a single brain
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    
    # save image here  
    image_path = os.path.join(output_dir, f'epoch_{epoch:03d}_step_{step:03d}_generated.png')
    img = reverse_transforms(image)
    img.save(image_path)


@torch.no_grad()
def sample_save_image(model, epoch, output_dir, device):
    # Sample noise
    img_size = IMAGE_SIZE
    img = torch.randn((1, 1, img_size, img_size), device=device)
    
    # Hyperparameters for step size and number of images to save per epoch
    num_images = 10
    stepsize = int(T/num_images)
    
    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(model, img, t)
        # Maintain natural range of distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            save_tensor_image(img.detach().cpu(), epoch, i, output_dir)
