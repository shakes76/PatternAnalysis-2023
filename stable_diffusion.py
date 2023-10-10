# ==== import from package ==== #
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange, repeat, reduce
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# ==== import from this folder ==== #
from model_VAE import VQVAE
from dataset import get_dataloader
from util import reset_dir, weight_scheduler, compact_large_image
from logger import Logger
from latent_dataset import get_latent_set
from model_diffusion import LatentDiffusionModel

DEVICE = torch.device("cuda")
print("DEVICE:", DEVICE)

mode = 'VQVAE'
load_epoch = 45

# Get latent set
latent_set = torch.load(f'collected_latents/{mode}_{load_epoch}.pt')
vae = torch.load(f'model_ckpt/{mode}/epoch_AE_{load_epoch}.pt')
vae.eval()

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

# Setup configs abou diffusion model
net = LatentDiffusionModel(in_channels=8, ch=32).to(DEVICE)
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

# Get dataloader
print("Latent shape:", normalized_latents.shape)
dataset = TensorDataset(normalized_latents, z_indices)
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

def test_epoch(epoch):
    with torch.no_grad():
        reset_dir(f'stable_diffusion_vis/epoch_{epoch}')
        # We only sample 6 images (and each images contain 32 idx)
        sample_n = 6
        # Scaling
        for cur_idx in range(sample_n):
            cond = torch.arange(0, 32, device=DEVICE, dtype=torch.long)
            sample_latent = net.sample_with_cond((1, 8, 16, 16), cond)
            sample_latent = sample_latent * latents_std + latents_mean
    
            # quant, diff_loss, ind = vae.quantize(sample_latent)
            sample_img = vae.decode(sample_latent, cond)
            
            sample_img = compact_large_image(sample_img.cpu(), HZ=4, WZ=8)
            
            plt.imsave(f'stable_diffusion_vis/epoch_{epoch}/{cur_idx}.png', sample_img[0] * 0.5 + 0.5, cmap='gray')

for epochs in range(150):
    total_loss, total_len = 0, 0
    for latent, zs in tqdm(dataloader, total=len(dataloader)):
        latent = latent.cuda()
        batch_size = latent.shape[0]

        optimizer.zero_grad()
        t = torch.randint(0, net.T, size=(batch_size, )).cuda()
        latent_noise, noise = net.get_noise(latent, t)
        predicted_noise = net.forward(latent_noise, t, zs)

        loss = torch.abs(noise.contiguous() - predicted_noise.contiguous())
        loss = torch.sum(loss) / batch_size
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_len += 1
    if epochs % 10 == 0:
        torch.save(net, f'model_ckpt/stable_diffusion/UNet_{epochs}.pt')
        test_epoch(epochs)
    print(f"epoch {epochs:4d}, loss {total_loss / total_len:.5f}")

