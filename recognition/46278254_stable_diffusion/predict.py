# ==== import from package ==== #
import torch
import imageio
import numpy as np
from einops import reduce
from tqdm import tqdm
# ==== import from this folder ==== #
from dataset import get_dataloader
from model_diffusion import LatentDiffusionModel
from model_VAE import VAE, VQVAE
from util import compact_large_image, ssim

# Fixed the random seed for reproducibiilty
torch.manual_seed(0)

DEVICE = torch.device("cuda")
print("DEVICE:", DEVICE)

mode = 'VQVAE'
load_epoch = 45
# Load VAE model
vae = torch.load(f'model_ckpt/{mode}/epoch_AE_{load_epoch}.pt')
vae.eval()

'''
Task 8: Use VQVAE to reconstruct image that over SSIM score 0.78
Task 9: Generate brain using stable diffusion. (This will generate gif.)
'''

task = 'TASK 8'
assert task in ['TASK 8', 'TASK 9']

if task == 'TASK 8':
    dataloader = get_dataloader('test', 32)
    with torch.no_grad():
        total_ssim = 0
        recon_imgs, brain_indices = [], []
        for now_step, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
            raw_img, seg_img, brain_idx, z_idx = [
                data.to(DEVICE) for data in batch_data]
            recon_img, regularization, latent = vae(raw_img, z_idx)

            # Record reconstructed images (for visualization) and brain indices (for labeling)
            recon_imgs.append(recon_img.detach().cpu())
            brain_indices.append(brain_idx.detach().cpu())

            # Calculate total ssim. (Window size is 11x11)
            total_ssim += ssim(raw_img * 0.5 + 0.5, recon_img *
                               0.5 + 0.5).item() * raw_img.shape[0]
    print(f"SSIM score: {total_ssim / len(dataloader.dataset)}")

elif task == 'TASK 9':
    load_DDPM_epoch = 300
    folder = f'visualize/ldm_{mode}_vis'

    # Get latent set
    latent_set = torch.load(f'collected_latents/{mode}_{load_epoch}.pt')

    net = torch.load(
        f'model_ckpt/stable_diffusion_{mode}/UNet_{load_DDPM_epoch}.pt')
    net.eval()

    # Try to normalize latents with conditions
    with torch.no_grad():
        latents, z_indices = latent_set.tensors[0], latent_set.tensors[1]
        if mode == 'VAE':
            # In 'VAE' mode, reparameterization latents into a space that decoder familiar with.
            # Note that Reparameterization has noise. (mean + exp(log_exp) * randn )
            # Therefore we can add different noise when training DDPM.
            latents, _ = vae.reparameterization(latents)

        # The distribution of latents we get is very chaotic, and its harmful to DDPM noise.
        # To avoid this issue, we'll normalize the latent into (-1, 1) before doing DDPM.
        normalized_latents = latents
        latents_mean = torch.zeros(
            [32, *latents.shape[1:]]).to(device=latents.device)
        latents_std = torch.zeros_like(latents_mean)
        for cur_z_idx in range(0, 32):
            chosen = z_indices == cur_z_idx

            # Note that mean & std is not mean & std. It's move and scaling factor.
            latents_mean[cur_z_idx] = reduce(
                latents[chosen], 'n c h w -> 1 c h w', 'min')
            latents_std[cur_z_idx] = reduce(
                latents[chosen], 'n c h w -> 1 c h w', 'max') - latents_mean[cur_z_idx]
            normalized_latents[chosen] = (
                latents[chosen] - latents_mean[cur_z_idx]) / latents_std[cur_z_idx]
            # Move range 0 ~ 1 to -1, 1
            normalized_latents[chosen] = 2 * (normalized_latents[chosen] - 0.5)

        # We only sample 6 images (and each images contain 32 idx)
        sample_n = 6
        for cur_idx in range(sample_n):
            # Define condition from 0 to 31 (z-index of brain).
            cond = torch.arange(0, 32, device=DEVICE, dtype=torch.long)
            # Get sample latent from network
            sample_latent = net.sample_with_cond(
                (1, latents.shape[1], 16, 16), cond, True)

            with imageio.get_writer(f'{folder}/ldm_{cur_idx}.gif', mode="I", fps=30) as writer:

                for idx, cur_sample in tqdm(enumerate(sample_latent), total=len(sample_latent)):
                    # We only collect image in 20x speed.
                    if idx % 20 != 0:
                        continue

                    # move -1 ~ 1) to 0 ~ 1
                    cur_sample = cur_sample * 0.5 + 0.5
                    # move 0 ~ 1 to original latent space
                    cur_sample = cur_sample.to(
                        DEVICE) * latents_std + latents_mean

                    # In VQVAE, we need quantize vector before decode.
                    if mode == 'VQVAE':
                        quant, diff_loss, ind = vae.quantize(cur_sample)
                        sample_img = vae.decode(quant, cond)
                    elif mode == 'VAE':
                        sample_img = vae.decode(cur_sample, cond)

                    sample_img = compact_large_image(
                        sample_img.cpu(), HZ=4, WZ=8)[0]

                    # To avoid alert, we cast float to uint8
                    sample_img = ((sample_img + 1) / 2 * 255).astype(np.uint8)

                    writer.append_data(sample_img)
                    if idx == len(sample_latent) - 1:
                        for _ in range(100 // 3):
                            writer.append_data(sample_img)
