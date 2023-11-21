"""
Generative Adversarial Network (GAN) Training Script

This script trains a Progressive Growing GAN (PGGAN) using the provided modules
for a given dataset. It implements both the generator and discriminator
training steps, including the calculation of the gradient penalty for the
Wasserstein GAN with Gradient Penalty (WGAN-GP) loss. The training is done
in a progressive manner, starting from a low resolution and gradually
increasing the image size.

@author: Yash Mittal
@ID: s48238690
"""

import torch
from torch import optim
import os
import matplotlib.pyplot as plt
import random
import numpy as np

import modules
import dataset

# Clear any previously cached data by Memory Persistence Service
torch.mps.empty_cache()

# Choose device ('mps' for Memory Persistence Service, 'cpu' if unavailable)
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Constants
latent_dim = 512
style_dim = 512
gradient_penalty_weight = 10
BATCH_SIZES = [256, 128, 64, 32, 16, 8]
progressive_epochs = [100] * len(BATCH_SIZES)
input_channels = 512
image_channels = 3
generator_learning_rate = 1e-3
discriminator_learning_rate = 5e-4
initial_image_size = 4

generator_losses = []  # List to store generator losses
discriminator_losses = []  # List to store discriminator losses

# Function to calculate gradient penalty
def calculate_gradient_penalty(discriminator, real_images, fake_images, interpolation_alpha, step, device):
    """
    Calculate the gradient penalty for enforcing the Lipschitz constraint.

    Args:
        discriminator (nn.Module): The discriminator (discriminator) network.
        real_images (torch.Tensor): real_images images.
        fake_images (torch.Tensor): Generated fake_images images.
        interpolation_alpha (float): A random value for the interpolation.
        step (int): The current step in progressive training.
        device (str): Device for computation ('mps' or 'cpu').

    Returns:
        torch.Tensor: The gradient penalty.

    """
    batch_size, _, _, _ = real_images.shape
    beta = torch.rand((batch_size, 1, 1, 1)).to(device)
    interpolated_images = real_images * beta + fake_images.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    mixed_scores = discriminator(interpolated_images, interpolation_alpha, step)

    gradients = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient_norm = gradients.view(gradients.shape[0], -1).norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

# Train function
def train_discriminator_and_generator(discriminator, gen, loader, step, interpolation_alpha, opt_discriminator, opt_gen):
    """
    Training loop for both the discriminator (discriminator) and the generator.

    Args:
        discriminator (nn.Module): The discriminator (discriminator) network.
        gen (nn.Module): The generator network.
        loader (DataLoader): DataLoader for training data.
        step (int): The current step in progressive training.
        interpolation_alpha (float): A value for controlling the fading of layers.
        opt_discriminator (optim.Optimizer): Optimizer for the discriminator.
        opt_gen (optim.Optimizer): Optimizer for the generator.

    Returns:
        float: Updated interpolation_alpha value for layer fading.

    """
    for batch_idx, real_images in enumerate(loader):
        real_images = real_images.to(DEVICE)
        cur_batch_size = real_images.shape[0]
        noise = torch.randn(cur_batch_size, latent_dim).to(DEVICE)
        fake_images = gen(noise, interpolation_alpha, step)

        discriminator_real_images = discriminator(real_images, interpolation_alpha, step)  # Pass step and interpolation_alpha to the discriminator
        discriminator_fake_images = discriminator(fake_images.detach(), interpolation_alpha, step)  # Pass step and interpolation_alpha to the discriminator
        gp = calculate_gradient_penalty(discriminator, real_images, fake_images, interpolation_alpha, step, DEVICE)
        # Pass step to the gradient penalty function

        loss_discriminator = (
                -(torch.mean(discriminator_real_images) - torch.mean(discriminator_fake_images))
                + gradient_penalty_weight * gp
                + (0.001) * torch.mean(discriminator_real_images ** 2)
        )
        discriminator_losses.append(abs(loss_discriminator.item()))
        discriminator.zero_grad()
        loss_discriminator.backward()
        opt_discriminator.step()

        gen_fake_images = discriminator(fake_images, interpolation_alpha, step)  # Pass step and interpolation_alpha to the discriminator
        loss_gen = -torch.mean(gen_fake_images)
        generator_losses.append(abs(loss_gen.item()))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        interpolation_alpha += cur_batch_size / (progressive_epochs[step] * 0.5 * len(loader.dataset))
        interpolation_alpha = min(interpolation_alpha, 1)

        print(
                f"Epoch [{current_epoch_label}/{total_epochs_per_step}] Batch Sample [{batch_idx + 1}/{len(loader)}] loss_gen: {loss_gen:.4f}"
            )

    return interpolation_alpha

# Model initialization
gen = modules.Generator(latent_dim, style_dim, input_channels, image_channels).to(DEVICE)
discriminator = modules.Discriminator(input_channels, image_channels).to(DEVICE)

# Optimizers
gen_params = [{'params': [param for name, param in gen.named_parameters() if 'map' not in name]}]
gen_params += [{'params': gen.style_mapping.parameters(), 'lr': 1e-5}]
opt_gen = optim.Adam(gen_params, lr=generator_learning_rate, betas=(0.0, 0.99))
opt_discriminator = optim.Adam(discriminator.parameters(), lr=discriminator_learning_rate, betas=(0.0, 0.99))

# Set training mode
gen.train()
discriminator.train()

# Progressive training
step = int(np.log2(initial_image_size / 4))
epoch_losses = []

for num_epochs in progressive_epochs[step:]:
    interpolation_alpha = 1e-7
    loader, data = dataset.get_data_loader(4 * 2 ** step)
    current_step_label = step + 1
    total_steps = len(progressive_epochs)
    print('Present image size: ' + str(4 * 2 ** step))
    print(f'Currently Running for Batch {current_step_label}/{total_steps}')

    for epoch in range(num_epochs):
        current_epoch_label = epoch + 1
        total_epochs_per_step = progressive_epochs[step]
        print("Starting next epoch\n")
        train_discriminator_and_generator(discriminator, gen, loader, step, interpolation_alpha, opt_discriminator, opt_gen)

    epoch_loss_gen = sum(generator_losses) / len(generator_losses)
    epoch_losses.append(epoch_loss_gen)

    step += 1
    generator_losses = []

# Save the models
torch.save(gen.state_dict(), 'Generator.pth')

# Plot the losses of generator and discriminator
def plot_losses(epoch_losses, discriminator_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses, label="Generator's Loss per Epoch", color="yellow")
    plt.plot(discriminator_losses, label="discriminator's Loss per Batch", color="green")
    plt.title("Loss Plot")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.show()

if not os.path.exists("output_images"):
    os.makedirs("output_images")
save_path = os.path.join("output_images", "LossPlot.png")
plot_losses(epoch_losses, discriminator_losses)