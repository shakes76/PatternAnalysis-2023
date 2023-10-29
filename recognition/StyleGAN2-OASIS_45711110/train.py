"Source code for training, validating, testing and saving the model"

import torch
import os
from torch import optim
from tqdm import tqdm
from torchvision.utils import save_image

from config import *
from dataset import get_data, get_loader
from modules import *

'''
This is an regularization penalty 
We try to reduce the L2 norm of gradients of the discriminator with respect to images.
'''
def gradient_penalty(critic, real, fake,device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)
 
    # Calculates the gradient of scores with respect to the images
    # and we need to create and retain graph since we have to compute gradients
    # with respect to weight on this loss.
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    # Reshape gradients to calculate the norm
    gradient = gradient.view(gradient.shape[0], -1)
    # Calculate the norm and then the loss
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty

# Samples z on random and fetches w from mapping network
def get_w(batch_size):

    z = torch.randn(batch_size, w_dim).to(device)
    w = mapping_network(z)
    # Expand w from the generator blocks
    return w[None, :, :].expand(log_resolution, -1, -1)

# Generates random noise for the generator block
def get_noise(batch_size):
    
    noise = []
    #noise res starts from 4x4
    resolution = 4

    # For each gen block
    for i in range(log_resolution):
        # First block uses 3x3 conv
        if i == 0:
            n1 = None
        # For rest of conv layer
        else:
            n1 = torch.randn(batch_size, 1, resolution, resolution, device=device)
            n2 = torch.randn(batch_size, 1, resolution, resolution, device=device)

        # add the noise tensors to the lsit
        noise.append((n1, n2))
        # subsequent block has 2x2 res
        resolution *= 2

    return noise

'''
Generate Imagees using the generator.
Images are saved in separate epoch folder with 100 images each

Epoch intervals is sent as parameter while the number of imgs and path is hard coded below.
'''
def generate_examples(gen, epoch, n=100):
    
    for i in range(n):
        with torch.no_grad():
            w     = get_w(1)
            noise = get_noise(1)
            img = gen(w, noise)
            if not os.path.exists(f'saved_examples/epoch{epoch}'):
                os.makedirs(f'saved_examples/epoch{epoch}')
            save_image(img*0.5+0.5, f"saved_examples/epoch{epoch}/img_{i}.png")

'''
Main training loop
'''
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

        # Use cuda AMP for accelerated training
        with torch.cuda.amp.autocast():
            # Generate fake image using the Generator
            fake = gen(w, noise)

            # Get a critic score for the fake and real image
            critic_fake = critic(fake.detach())
            critic_real = critic(real)

            # Calculate and log gradient penalty
            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + lambda_gp * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )

        '''
        Reset gradients for the Discriminator
        Backpropagate the loss and update the discriminator's weights
        '''
        critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()

        # Get score for the Discriminator and the Generator
        gen_fake = critic(fake)
        loss_gen = -torch.mean(gen_fake)

        # Apply path length penalty on every 16 Batches
        if batch_idx % 16 == 0:
            plp = path_length_penalty(w, fake)
            if not torch.isnan(plp):
                loss_gen = loss_gen + plp

        '''
        Reset gradients for the mapping network and the generator
        Backpropagate the generator loss
        Update generator's weights and Update mapping network's weights
        '''
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
print('Device: ', device)

# Module initilization
loader              = get_data()

gen                 = Generator(log_resolution, w_dim).to(device)
critic              = Discriminator(log_resolution).to(device)
mapping_network     = MappingNetwork(z_dim, w_dim).to(device)
path_length_penalty = PathLengthPenalty(0.99).to(device)

# Initilise Adam optimiser
opt_gen             = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.0, 0.99))
opt_critic          = optim.Adam(critic.parameters(), lr=learning_rate, betas=(0.0, 0.99))
opt_mapping_network = optim.Adam(mapping_network.parameters(), lr=learning_rate, betas=(0.0, 0.99))

# Train the following modules
gen.train()
critic.train()
mapping_network.train()

# loop over total epcoh.
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
    # Save generator's fake image on every 50th epoch
    if epoch % 50 == 0:
    	generate_examples(gen, epoch)
