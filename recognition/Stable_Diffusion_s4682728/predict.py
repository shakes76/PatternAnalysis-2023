from imports import *
from dataset import *
from modules import *
from utils import *

@torch.no_grad()
def reverse_diffusion_step(model, x, t, t_index):
    '''
    Sourced from: https://huggingface.co/blog/annotated-diffusion
    Author: Niels Rogge, Kashif Rasul
    '''
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    # use our model to predict the mean
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:
        posterior_var_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_var_t) * noise

@torch.no_grad()
def image_generation(model, shape=(1, 1, 256, 256), save_path=None):
    device = next(model.parameters()).device

    # Setup figure for plotting
    fig = plt.figure(figsize=(15,15))
    fig.patch.set_facecolor('black')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis("off")

    # Define reverse transformation for image normalization
    reverse_transform = Compose([
                    Lambda(lambda t: (t + 1) / 2),
                    Lambda(lambda t: t * 255.),
                ])

    # Define grid dimensions for subplot (3x3 grid)
    rows = 3**2 
    cols = 3
    stepsize = int(timesteps/1)
    counter = 1

    for i in range(1, rows+1):
        # Initialize image with random noise
        img = torch.randn(shape, device=device)
        for j in reversed(range(0, timesteps)):
            t = torch.full((1,), j, device=device, dtype=torch.long)
            with torch.no_grad():
                # Perform a reverse diffusion step
                img = reverse_diffusion_step(model, img, t, j)
            # Plot the reconstructed image at specified timesteps
            if j % stepsize == 0:
                ax = plt.subplot(int(math.sqrt(rows)), cols, counter)
                ax.axis("off")
                plt.imshow(reverse_transform(img[0].permute(1,2,0).detach().cpu()), cmap="gray")
                counter+=1
    
    # Save the image
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  
