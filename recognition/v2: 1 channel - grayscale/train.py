"Source code for training, validating, testing and saving the model"

import torch
import os

from torchvision.utils import save_image
from torch import optim
from tqdm import tqdm

from config import *
from dataset import get_data
from v2.modules import *

# Function to generate latent vectors 'w' from random noise
def get_w(batch_size):
    # Generate 'w' from random noise
    z = torch.randn(batch_size, w_dim).to(device)
    w = mapping_network(z)
    return w[None, :, :].expand(log_resolution, -1, -1)

# Function to generate noise inputs for the generator
def get_noise(batch_size):
    # Generate noise inputs for the generator
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

def generate_examples(gen, epoch, n=100):

    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            w     = get_w(1)
            noise = get_noise(1)
            img = gen(w, noise)
            if not os.path.exists(f'saved_examples/epoch{epoch}'):
                os.makedirs(f'saved_examples/epoch{epoch}')
            save_image(img*0.5+0.5, f"saved_examples/epoch{epoch}/img_{i}.png")
    #print()
    #print("generated example {epoch}")
    #print()
    #print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

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
        #print("-xx-xx-x-x-x-x-x-xXXXXXXXXXXXXXXXXXXXXXXXXloop tqdmXXXXXXXXXXXXXXXXXXXXXXXXX--x-x-", flush=True)
        real = real.to(device)
        #print(real.shape)
        cur_batch_size = real.shape[0]
        #print(cur_batch_size)
        #print("-xx-xx-x-x-x-x-x-xx-x--xxxxxxxxxXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxyy-x-xx-x--x-xxxxxx--x-x-", flush=True)
        

        w     = get_w(cur_batch_size)
        noise = get_noise(cur_batch_size)
        with torch.cuda.amp.autocast():
            #print("-xx-xx-x-x-x-x-x-xx-x--x-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx-xxxxxx--x-x-", flush=True)
            fake = gen(w, noise)
            #print("-xx-xx-x-x-x-x-x-xx-x--x-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx-xxxxxx--x-x-", flush=True)
            critic_fake = critic(fake.detach())
            #print("-xx-xx-x-x-x-x-x-xx-x--x-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx-xxxxxx--x-x-", flush=True)
            #print(critic_fake)
            
            critic_real = critic(real)
            #print("-xx-xx-x-x-x-x-x-xx-x--x-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx-xxxxxx--x-x-", flush=True)
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


# Device Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print('Device: ', device)

loader              = get_data(DATA, log_resolution, batch_size)

gen                 = Generator(log_resolution, w_dim).to(device)
critic              = Discriminator(log_resolution).to(device)
mapping_network     = MappingNetwork(z_dim, w_dim).to(device)
path_length_penalty = PathLengthPenalty(0.99).to(device)

opt_gen             = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.0, 0.99))
opt_critic          = optim.Adam(critic.parameters(), lr=learning_rate, betas=(0.0, 0.99))
opt_mapping_network = optim.Adam(mapping_network.parameters(), lr=learning_rate, betas=(0.0, 0.99))

#print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
#print("training: Generator")
gen.train()
#print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
#print("training: Discriminator")
critic.train()
#print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
#print("training: MappingNetwork")
mapping_network.train()
#print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")


for epoch in range(epochs):
    #print()
    #print("----------------------------------------------")
    #print("in main train_fn loop")
    train_fn(
        critic,
        gen,
        path_length_penalty,
        loader,
        opt_critic,
        opt_gen,
        opt_mapping_network,
    )
    print()
    print("----------------------------------------------")
    print(epoch)
    print()
    print("----------------------------------------------")
    if epoch % 10 == 0:
        #print("in gen ex loop")
        generate_examples(gen, epoch)
