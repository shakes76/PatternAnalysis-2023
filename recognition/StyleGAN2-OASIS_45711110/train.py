"Source code for training, validating, testing and saving the model"

import torch
import os
from torch import optim
from tqdm import tqdm
from torchvision.utils import save_image

from config import *
from dataset import get_data, get_loader
from modules import *

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

# fetches w from mapping network
def get_w(batch_size):

    z = torch.randn(batch_size, w_dim).to(device)
    w = mapping_network(z)
    return w[None, :, :].expand(log_resolution, -1, -1)

# Generates random noise
def get_noise(batch_size):
    
    noise = []
    resolution = 4

    for i in range(log_resolution):
        if i == 0:
            n1 = None
        else:
            n1 = torch.randn(batch_size, 1, resolution, resolution, device=device)
        n2 = torch.randn(batch_size, 1, resolution, resolution, device=device)

        noise.append((n1, n2))
        resolution *= 2

    return noise

def generate_examples(gen, epoch, n=100):
    
    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            w     = get_w(1)
            noise = get_noise(1)
            img = gen(w, noise)
            if not os.path.exists(f'saved_examples/epoch{epoch}'):
                os.makedirs(f'saved_examples/epoch{epoch}')
            save_image(img*0.5+0.5, f"saved_examples/epoch{epoch}/img_{i}.png")

    gen.train()

def train_fn(
    critic,
    gen,
    path_length_penalty,
    loader,
    opt_critic,
    opt_gen,
    opt_mapping_network,
):
    loop = tqdm(loader, leave=True)

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        w     = get_w(cur_batch_size)
        noise = get_noise(cur_batch_size)

        with torch.cuda.amp.autocast():
            fake = gen(w, noise)
            critic_fake = critic(fake.detach())
            
            critic_real = critic(real)
            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + lambda_gp * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )

        critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()

        gen_fake = critic(fake)
        loss_gen = -torch.mean(gen_fake)

        if batch_idx % 16 == 0:
            plp = path_length_penalty(w, fake)
            if not torch.isnan(plp):
                loss_gen = loss_gen + plp

        mapping_network.zero_grad()
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        opt_mapping_network.step()

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
        )

loader              = get_loader()

gen                 = Generator(log_resolution, w_dim).to(device)
critic              = Discriminator(log_resolution).to(device)
mapping_network     = MappingNetwork(z_dim, w_dim).to(device)
path_length_penalty = PathLengthPenalty(0.99).to(device)

opt_gen             = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.0, 0.99))
opt_critic          = optim.Adam(critic.parameters(), lr=learning_rate, betas=(0.0, 0.99))
opt_mapping_network = optim.Adam(mapping_network.parameters(), lr=learning_rate, betas=(0.0, 0.99))

gen.train()
critic.train()
mapping_network.train()


for epoch in range(epochs):
    train_fn(
        critic,
        gen,
        path_length_penalty,
        loader,
        opt_critic,
        opt_gen,
        opt_mapping_network,
    )
    if epoch % 50 == 0:
    	generate_examples(gen, epoch)
