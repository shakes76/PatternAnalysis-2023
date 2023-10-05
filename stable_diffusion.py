# ==== import from package ==== #
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange, repeat, reduce
from collections import defaultdict
# ==== import from this folder ==== #
from model_VQVAE import VQVAE
from discriminator import NLayerDiscriminator, weights_init
from dataset import get_dataloader
from util import reset_dir, weight_scheduler, compact_large_image
from logger import Logger
DEVICE = torch.device("cuda")
print("DEVICE:", DEVICE)

batch_size = 6
dataloader = get_dataloader(mode='train_and_validate', batch_size=batch_size)

vae = VQVAE().to(DEVICE)
vae = torch.load(f'model_ckpt/VQVAE/epoch_AE_50.pt')
vae.eval()
print()

def get_latents(net, dataloader):
    latents = []
    for now_step, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
        raw_img, seg_img, brain_idx, z_idx = [
            data.to(DEVICE) for data in batch_data]
        with torch.no_grad():
            batch_size = raw_img.shape[0]
            latent = net.encode(raw_img)
            latents.append(latent.detach())
    latents = torch.cat(latents)
    return latents

with torch.no_grad():
    latents = get_latents(vae, dataloader).detach()
from model_diffusion import LatentDiffusionModel
net = LatentDiffusionModel(in_channels=8, ch=32).to(DEVICE)
from torchinfo import summary
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

from torch.utils.data import TensorDataset, DataLoader

print("Latent shape:", latents.shape)
from einops import reduce
with torch.no_grad():
    latents_mean = torch.mean(latents, dim=0, keepdim=True)
    latents_std = torch.std(latents, dim=0, keepdim=True)
    latents = (latents - latents_mean) / latents_std
    latents = latents.detach()
    print(latents_mean.shape, latents_std.shape)
    # print(latents_mean, latents_std)
dataset = TensorDataset(latents)
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
from tqdm import tqdm
# net = torch.load('stable_diffusion.pt')
# net = torch.load('diffusion_MNIST.pt')
for epochs in range(200):
    total_loss, total_len = 0, 0
    for latent, in tqdm(dataloader, total=len(dataloader)):
        latent = latent.cuda()
        batch_size = latent.shape[0]

        optimizer.zero_grad()
        t = torch.randint(0, net.T, size=(batch_size, )).cuda()
        latent_noise, noise = net.get_noise(latent, t)
        predicted_noise = net.forward(latent_noise, t)

        loss = torch.abs(noise.contiguous() - predicted_noise.contiguous())
        loss = torch.sum(loss) / batch_size
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_len += 1
    torch.save(net, 'stable_diffusion.pt')
    print(f"epoch {epochs:4d}, loss {total_loss / total_len:.5f}")

with torch.no_grad():
    sample_latent = net.sample((32, 8, 16, 16))
    # Scaling
    sample_latent = sample_latent * latents_std + latents_mean
 
    quant, diff_loss, (_, _, ind) = vae.quantize(sample_latent)
    sample_img = vae.decode(quant)
    
    sample_img = compact_large_image(sample_img.cpu(), HZ=4, WZ=8)
    plt.imsave('stable_diffusion.png', sample_img[0] * 0.5 + 0.5, cmap='gray')
