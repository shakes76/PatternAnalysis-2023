import torch
from model_diffusion import LatentDiffusionModel
from model_VAE import VAE, VQVAE
from util import compact_large_image
import matplotlib.pyplot as plt
import imageio
import numpy as np
from tqdm import tqdm

DEVICE = torch.device("cuda")
print("DEVICE:", DEVICE)

load_epoch = 45
mode = 'VQVAE'

# Get latent set
latent_set = torch.load(f'collected_latents/{mode}_{load_epoch}.pt')
vae = torch.load(f'model_ckpt/{mode}/epoch_AE_{load_epoch}.pt')
vae.eval()

net = torch.load(f'model_ckpt/stable_diffusion/UNet_140.pt')
net.eval()



# Try to normalize latents with conditions
with torch.no_grad():
    latents, z_indices = latent_set.tensors[0], latent_set.tensors[1]
    normalized_latents = latents
    latents_mean = torch.zeros([32, *latents.shape[1:]]).to(device=latents.device)
    latents_std = torch.zeros_like(latents_mean)
    for cur_z_idx in range(0, 32):
        chosen = z_indices == cur_z_idx
        from einops import reduce
        latents_mean[cur_z_idx] = reduce(latents[chosen], 'n c h w -> 1 c h w', 'min')
        latents_std[cur_z_idx] = reduce(latents[chosen], 'n c h w -> 1 c h w', 'max') - latents_mean[cur_z_idx]
        normalized_latents[chosen] = (latents[chosen] - latents_mean[cur_z_idx]) / latents_std[cur_z_idx]
    print(normalized_latents.shape)

    # We only sample 6 images (and each images contain 32 idx)
    sample_n = 1
    # Scaling
    for cur_idx in range(sample_n):
        cond = torch.arange(0, 32, device=DEVICE, dtype=torch.long)
        sample_latent = net.sample_with_cond((1, 8, 16, 16), cond, True)

        with imageio.get_writer(f'ldm_{sample_n}.gif', mode="I",fps=30) as writer:
            for idx, cur_sample in tqdm(enumerate(sample_latent), total=len(sample_latent)):
                cur_sample = cur_sample.to(DEVICE)
                cur_sample = cur_sample * latents_std + latents_mean
                quant, diff_loss, ind = vae.quantize(cur_sample)
                sample_img = vae.decode(quant, cond)
                
                sample_img = compact_large_image(sample_img.cpu(), HZ=4, WZ=8)[0]
                sample_img = ( (sample_img + 1) / 2  * 255).astype(np.uint8)
                
                
                
                writer.append_data(sample_img)
                if idx == len(sample_latent) - 1:
                    print("write end")
                    for _ in range(100 // 3):
                        writer.append_data(sample_img)

        # plt.imsave(f'stable_diffusion_vis/epoch_{epoch}/{cur_idx}.png', sample_img[0] * 0.5 + 0.5, cmap='gray')
