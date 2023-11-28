#Source code for training the StyleGAN2

import torch
from torch import optim
from tqdm import tqdm

import predict
from config import *
from dataset import get_data
from modules import *
from utils import get_w, get_noise, gradient_penalty

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

    curr_Gloss = []
    curr_Dloss = []

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        w     = get_w(cur_batch_size, mapping_network, device)
        noise = get_noise(cur_batch_size, device)

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
        
        # Append the observed Discriminator loss to the list
        curr_Dloss.append(loss_critic.item())

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

        # Append the observed Generator loss to the list
        curr_Gloss.append(loss_gen.item())

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

    return (curr_Dloss, curr_Gloss)
 
if __name__ == "__main__":

    # Device Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

    # Module initilization
    loader              = get_data(DATA, log_resolution, batch_size)

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

    # Keeps a Log of total loss over the training
    G_Loss = []
    D_Loss = []

    # loop over total epcoh.
    for epoch in range(epochs):
        curr_Gloss, curr_Dloss = train_fn(
            critic,
            gen,
            path_length_penalty,
            loader,
            opt_critic,
            opt_gen,
            opt_mapping_network,
        )

        # Append the current loss to the main list
        G_Loss.extend(curr_Gloss)
        D_Loss.extend(curr_Dloss)

        # Save generator's fake image on every 50th epoch
        if epoch % 10 == 0:

            predict.generate_examples(gen, mapping_network, epoch, device)

    predict.plot_loss(G_Loss, D_Loss)
