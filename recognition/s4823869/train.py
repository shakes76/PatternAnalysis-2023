import torch
from torch import optim
from tqdm import tqdm
import os
from math import log2
import matplotlib.pyplot as plt

import modules
import dataset

# Release GPU memory (MPS environment)
torch.mps.empty_cache()

# Choose device ('mps' for Memory Persistence Service, 'cpu' if unavailable)
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
Z_DIM = 512
W_DIM = 512
LAMBDA_GP = 10
BATCH_SIZES = [256, 128, 64, 32, 16, 8]
PROGRESSIVE_EPOCHS = [1] * (len(BATCH_SIZES)-5)  # Reduced to 5 for faster training
IN_CHANNELS = 512
CHANNELS_IMG = 3
LR = 1e-3
LR_CRITIC = 5e-4
START_TRAIN_IMG_SIZE = 4

# Regularization on the discriminator / critic
def gradient_penalty(critic, real, fake, alpha, train_step, device="mps"):
    """
    Calculate the gradient penalty for improved Wasserstein GAN training.

    Args:
        critic (nn.Module): The critic/discriminator model.
        real (torch.Tensor): Real images.
        fake (torch.Tensor): Generated fake images.
        alpha (float): Interpolation factor.
        train_step (int): Training step.
        device (str): Device for calculations ('mps' or 'cpu').

    Returns:
        torch.Tensor: Calculated gradient penalty.
    """
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, train_step)

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

gen_losses = []  # Generator losses
critic_losses = []  # Critic losses

# Training function
def train_fn(critic, gen, loader, dataset, step, alpha, opt_critic, opt_gen):
    """
    Perform a training epoch for the generator and discriminator/critic.

    Args:
        critic (nn.Module): The critic/discriminator model.
        gen (nn.Module): The generator model.
        loader (DataLoader): Data loader for the training dataset.
        dataset (Dataset): Training dataset.
        step (int): Current step in progressive training.
        alpha (float): Interpolation factor.
        opt_critic (torch.optim.Optimizer): Optimizer for the critic.
        opt_gen (torch.optim.Optimizer): Optimizer for the generator.

    Returns:
        float: Updated alpha value.
    """
    loop = tqdm(loader, leave=True)

    for batch_idx, real in enumerate(loop):
        real = real.to(DEVICE)
        cur_batch_size = real.shape[0]
        noise = torch.randn(cur_batch_size, Z_DIM).to(DEVICE)  # z
        fake = gen(noise, alpha, step)
        critic_real = critic(real, alpha, step)
        critic_fake = critic(fake.detach(), alpha, step)
        gp = gradient_penalty(critic, real, fake, alpha, step, DEVICE)

        # Enhanced version of the WGAN discriminator loss
        loss_critic = (
            -(torch.mean(critic_real) - torch.mean(critic_fake))
            + LAMBDA_GP * gp
            + (0.001) * torch.mean(critic_real ** 2)
        )
        # Store the absolute loss for plotting
        critic_losses.append(abs(loss_critic.item()))

        critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()

        gen_fake = critic(fake, alpha, step)
        loss_gen = -torch.mean(gen_fake)
        # Store the absolute loss for plotting
        gen_losses.append(abs(loss_gen.item()))

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        alpha += cur_batch_size / (PROGRESSIVE_EPOCHS[step] * 0.5 * len(dataset))
        alpha = min(alpha, 1)

        # Display both losses during training
        loop.set_postfix(
            loss_critic=loss_critic.item(),
            loss_gen=loss_gen.item()
        )
    return alpha

# Model initialization
gen = modules.Generator(Z_DIM, W_DIM, IN_CHANNELS, CHANNELS_IMG).to(DEVICE)
critic = modules.Discriminator(IN_CHANNELS, CHANNELS_IMG).to(DEVICE)

# Optimization
opt_gen = optim.Adam([{'params': [param for name, param in gen.named_parameters() if 'map' not in name]},
                      {'params': gen.map.parameters(), 'lr': 1e-5}], lr=LR, betas=(0.0, 0.99))
opt_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC, betas=(0.0, 0.99))

# Train mode
gen.train()
critic.train()

# Progressive training
step = int(log2(START_TRAIN_IMG_SIZE / 4))
for num_epochs in PROGRESSIVE_EPOCHS[step:]:
    alpha = 1e-7
    loader, data = dataset.get_loader(4 * 2 ** step)
    print('Current image size: ' + str(4 * 2 ** step))

    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        alpha = train_fn(critic, gen, loader, data, step, alpha, opt_critic, opt_gen)

    step += 1

# Save the models
torch.save(gen.state_dict(), 'Generator.pth')

# Plot the losses of generator and discriminator
def plot_and_save_losses(gen_losses, critic_losses, save_path):
    """
    Plot and save the generator and critic losses to a file.

    Args:
        gen_losses (list): List of generator losses.
        critic_losses (list): List of critic losses.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(gen_losses, label="Generator Loss", color="yellow")
    plt.plot(critic_losses, label="Critic/Discriminator Loss", color="green")
    plt.title("Training Losses")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()

    # Save the plot to a file
    plt.savefig(save_path)
    plt.close()

save_path = os.path.join("output_images", "losses_plot.png")
plot_and_save_losses(gen_losses, critic_losses, save_path)
