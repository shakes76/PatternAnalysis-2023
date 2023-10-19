import modules, os, torch, math
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from torchvision.utils import save_image

#Dataset utils

# Path to data images
root_path = 'data/keras_png_slices_data'

IMAGE_SIZE = 256 # Power of 2
BATCH_SIZE = 32

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

T = 100
betas = quadratic_beta_schedule(timesteps=T)

# Pre-calculate terms for closed form equation
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

#Train utils

# Output directory to save images
output_dir = './generated_images'

# Number of Epochs for training
epochs = T

# Loss function
def get_loss(model, x_0, t, device):
    x_noisy, noise = modules.forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

# Sampling
@torch.no_grad()
def sample_timestep(model, x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    #convert to greyscale for 1 channel input
    x = transforms.functional.rgb_to_grayscale(x)
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

# Save output images
def save_tensor_image(image, epoch, step, output_dir, start_time):
    
    """
    Plots image after applying reverse transformations.
    """

    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    if len(image.shape) == 4:
        image = image[0, :, :, :]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img_path = f"{output_dir}/generated_epoch_{epoch:03d}_step_{step:03d}_{start_time}.png"
    img = reverse_transforms(image)
    img.save(img_path)


@torch.no_grad()
def sample_save_image(model, epoch, output_dir, device, start_time):
    # Sample noise
    img_size = IMAGE_SIZE
    img = torch.randn((1, 1, img_size, img_size), device=device)
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(model, img, t)
        # Maintain natural range of distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            save_tensor_image(img.detach().cpu(), epoch, i, output_dir, start_time)

#style gan

DEVICE                  = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE           = 1e-3
LOG_RESOLUTION          = int(math.log2(IMAGE_SIZE))
Z_DIM                   = IMAGE_SIZE
W_DIM                   = IMAGE_SIZE
LAMBDA_GP               = 10

def gradient_penalty(critic, real, fake,device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)
 
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def get_w(mapping_network, batch_size):
    z = torch.randn(batch_size, W_DIM).to(DEVICE)
    w = mapping_network(z)
    return w[None, :, :].expand(LOG_RESOLUTION, -1, -1)

def get_noise(batch_size):
        noise = []
        resolution = 4

        for i in range(LOG_RESOLUTION):
            if i == 0:
                n1 = None
            else:
                n1 = torch.randn(batch_size, 1, resolution, resolution, device=DEVICE)
            n2 = torch.randn(batch_size, 1, resolution, resolution, device=DEVICE)

            noise.append((n1, n2))

            resolution *= 2

        return noise
        
def generate_examples(mapping_network, gen, epoch, start_time, n=100):
    for i in range(n):
        img = generate_images(mapping_network, gen)
        if not os.path.exists(f'saved_examples/{start_time}_epoch{epoch}'):
            os.makedirs(f'saved_examples/{start_time}_epoch{epoch}')
        save_image(img, f"saved_examples/{start_time}_epoch{epoch}/img_{i}.png")

def generate_images(mapping_network, gen, n=1):
    gen.eval()
    with torch.no_grad():
        w     = get_w(mapping_network, n)
        noise = get_noise(n)
        img   = gen(w, noise)
    gen.train()
    return img*0.5+0.5