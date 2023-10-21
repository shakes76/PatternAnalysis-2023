from imports import *
from dataset import *
from modules import *
from utils import *

"""
Taken from https://huggingface.co/blog/annotated-diffusion
"""
@torch.no_grad()
def reverse_diffusion_step(model, x, t, t_index):
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
def image_reconstruction(model, shape=(1, 1, 256, 256), save_path=None):
    device = next(model.parameters()).device

    fig = plt.figure(figsize=(15,15))
    fig.patch.set_facecolor('black')
    plt.axis("off")

    reverse_transform = Compose([
                    Lambda(lambda t: (t + 1) / 2),
                    Lambda(lambda t: t * 255.),
                ])

    img = torch.randn(shape, device=device)

    rows = 3**2
    cols = 3
    stepsize = int(timesteps/1)
    counter = 1

    for i in range(1, rows+1):
        img = torch.randn((1,1,256,256)).cuda()
        for j in reversed(range(0, timesteps)):
            t = torch.full((1,), j, device=device, dtype=torch.long)
            with torch.no_grad():
                img = reverse_diffusion_step(model, img, t, j)
            if j % stepsize == 0:
                ax = plt.subplot(int(math.sqrt(rows)), cols, counter)
                ax.axis("off")
                plt.imshow(reverse_transform(img[0].permute(1,2,0).detach().cpu()), cmap="gray")
                counter+=1
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  
