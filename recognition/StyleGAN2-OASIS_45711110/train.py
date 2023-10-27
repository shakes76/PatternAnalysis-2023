"Source code for training, validating, testing and saving the model"

import torch
from torch import optim
import tqdm

from config import z_dim, w_dim, device, lambda_gp, learning_rate, log_resolution, epochs
from dataset import get_data
from modules import MappingNetwork, Generator, Discriminator, PathLengthPenalty
from predict import generate_examples
from utils import get_w, get_noise, gradient_penalty

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

        MappingNetwork.zero_grad()
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        opt_mapping_network.step()

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
        )

loader              = get_data()

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