from imports import *
from dataset import *
from modules import *
from utils import *
from train import epochs

# Load trained model
model_path = f"diffusion_network{epochs}.pth"
model = DiffusionNetwork()
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

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
def reverse_diffusion(model, shape=(1, 1, 256, 256)):
    device = next(model.parameters()).device
    fig = plt.figure(figsize=(15, 15))
    plt.axis("off")

    reverse_transform = Compose([
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t * 255.),
    ])

    rows = 3
    cols = 3
    counter = 1

    for i in range(1, rows * cols + 1):
        img = torch.randn((1, 1, 256, 256)).cuda()
        t = torch.full((1,), 0, device=device, dtype=torch.long)  # Assuming the last time step
        with torch.no_grad():
            img = reverse_diffusion_step(model, img, t, 0)  # Assuming `reverse_diffusion_step` is defined elsewhere
        ax = plt.subplot(rows, cols, counter)
        ax.axis("off")
        plt.imshow(reverse_transform(img[0].permute(1, 2, 0).detach().cpu()), cmap="gray")
        counter += 1
    
    save_dir = os.path.expanduser("~/demo_eiji/sd/images")
    full_path = os.path.join(save_dir, "image_visualization.png")
    plt.savefig(full_path)

